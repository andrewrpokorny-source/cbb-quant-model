"""
Auto-settlement engine for betting_history.csv.

Matches pending bets to ESPN completed game scores, determines results,
and updates the CSV with payout/profit calculations for:
- spread bets (standard and Kalshi YES/NO thresholds)
- game-winner bets (ML / GAME YES/NO)

Usage:
    python settle_bets.py --league mens|womens
"""

import argparse
import os
import re
import logging
import pandas as pd
from datetime import datetime, timedelta
from difflib import get_close_matches
import pytz

from grade_predictions import fetch_completed_games

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "betting_history.csv")


def parse_bet_line(line_str):
    """
    Parse a bet line string into components.

    Examples:
        "Providence +15.5"      -> {team: "Providence", spread: 15.5, side: None, bet_type: "spread"}
        "UConn -15.5 NO"        -> {team: "UConn", spread: -15.5, side: "NO", bet_type: "spread"}
        "Furman -10.5 YES"      -> {team: "Furman", spread: -10.5, side: "YES", bet_type: "spread"}
        "South Carolina ML YES" -> {team: "South Carolina", spread: None, side: "YES", bet_type: "game"}
        "UConn ML"              -> {team: "UConn", spread: None, side: None, bet_type: "game"}
    """
    line_str = str(line_str).strip()

    # Handle embedded newlines from garbled OCR data:
    # search from the end for a line matching known bet patterns
    if "\n" in line_str:
        spread_re = re.compile(r"^(.+?)\s+([+-]?\d+\.?\d*)(?:\s+(YES|NO))?$", re.IGNORECASE)
        game_re = re.compile(r"^(.+?)\s+(?:ML|MONEYLINE|GAME)(?:\s+(YES|NO))?$", re.IGNORECASE)
        for part in reversed(line_str.split("\n")):
            part = part.strip()
            if spread_re.match(part) or game_re.match(part):
                line_str = part
                break
        else:
            # No line matched known patterns; use the last non-empty line.
            parts = [p.strip() for p in line_str.split("\n") if p.strip()]
            line_str = parts[-1] if parts else line_str

    # Check for Kalshi YES/NO suffix
    side = None
    if line_str.endswith(" YES"):
        side = "YES"
        line_str = line_str[:-4].strip()
    elif line_str.endswith(" NO"):
        side = "NO"
        line_str = line_str[:-3].strip()

    # GAME moneyline format (Kalshi GAME contracts and standard ML).
    game_match = re.match(r"^(.+?)\s+(?:ML|MONEYLINE|GAME)$", line_str, re.IGNORECASE)
    if game_match:
        team = game_match.group(1).strip()
        return {"team": team, "spread": None, "side": side, "bet_type": "game"}

    # Split off the spread (last token should be +/- number)
    match = re.match(r"^(.+?)\s+([+-]?\d+\.?\d*)$", line_str)
    if not match:
        # Support explicit YES/NO game lines without "ML" suffix.
        if side in {"YES", "NO"} and line_str:
            return {"team": line_str, "spread": None, "side": side, "bet_type": "game"}
        return None

    team = match.group(1).strip()
    spread = float(match.group(2))

    return {"team": team, "spread": spread, "side": side, "bet_type": "spread"}


def match_bet_by_team(line_str, completed_games):
    """Fallback: match a bet to a game using just the team name from the line field."""
    parsed = parse_bet_line(line_str)
    if not parsed:
        return None

    team = parsed["team"]
    for (home, away), result in completed_games.items():
        if _team_matches(team, home) or _team_matches(team, away):
            return result

    return None


def match_bet_to_game(game_str, completed_games):
    """
    Match a bet's game string (e.g. "Providence vs UConn") to ESPN completed games.

    completed_games: dict from fetch_completed_games() keyed by (home_name, away_name)

    Returns the game result dict or None.
    """
    if not game_str or (isinstance(game_str, float) and pd.isna(game_str)):
        return None

    # Parse "Away vs Home" or "Away @ Home" format
    for sep in [" vs ", " @ "]:
        if sep in str(game_str):
            parts = str(game_str).split(sep)
            if len(parts) == 2:
                team_a, team_b = parts[0].strip(), parts[1].strip()
                break
    else:
        return None

    # Try exact match (team_a=away, team_b=home)
    for (home, away), result in completed_games.items():
        if (home == team_b and away == team_a) or (home == team_a and away == team_b):
            return result

    # Try substring match
    for (home, away), result in completed_games.items():
        a_matches = team_a in home or team_a in away or home in team_a or away in team_a
        b_matches = team_b in home or team_b in away or home in team_b or away in team_b
        if a_matches and b_matches:
            return result

    # Fuzzy match as last resort
    all_team_names = []
    for (home, away) in completed_games.keys():
        all_team_names.extend([home, away])

    match_a = get_close_matches(team_a, all_team_names, n=1, cutoff=0.6)
    match_b = get_close_matches(team_b, all_team_names, n=1, cutoff=0.6)

    if match_a and match_b:
        matched_a = match_a[0]
        matched_b = match_b[0]
        for (home, away), result in completed_games.items():
            if {matched_a, matched_b} == {home, away}:
                return result

    return None


def determine_bet_result(parsed_line, game_result, game_str):
    """
    Determine if a bet won, lost, or pushed.

    parsed_line: output of parse_bet_line()
    game_result: dict with home_score, away_score, home_name, away_name
    game_str: original game string like "Providence vs UConn"

    Returns: "win", "loss", or "void"
    """
    team = parsed_line["team"]
    spread = parsed_line["spread"]
    side = parsed_line["side"]
    bet_type = parsed_line.get("bet_type", "spread")

    home_score = game_result["home_score"]
    away_score = game_result["away_score"]
    home_name = game_result["home_name"]
    away_name = game_result["away_name"]

    # Figure out which team was picked
    picked_home = _team_matches(team, home_name)
    picked_away = _team_matches(team, away_name)

    if not picked_home and not picked_away:
        return None

    if bet_type == "game":
        # Winner market:
        # - side YES or no side: team must win
        # - side NO: team must lose
        if home_score == away_score:
            return "void"
        team_won = (picked_home and home_score > away_score) or (picked_away and away_score > home_score)
        if side == "NO":
            return "loss" if team_won else "win"
        return "win" if team_won else "loss"

    if side is not None:
        # Kalshi bet: the spread is a threshold for the named team's margin
        # "UConn -15.5 NO" means: bet that UConn does NOT win by more than 15.5
        # "Furman -10.5 YES" means: bet that Furman DOES win by more than 10.5

        if picked_home:
            team_margin = home_score - away_score
        else:
            team_margin = away_score - home_score

        threshold = abs(spread)

        if side == "YES":
            # YES = team wins by more than threshold
            if team_margin > threshold:
                return "win"
            elif team_margin == threshold:
                return "void"
            else:
                return "loss"
        else:  # NO
            # NO = team does NOT win by more than threshold
            if team_margin < threshold:
                return "win"
            elif team_margin == threshold:
                return "void"
            else:
                return "loss"
    else:
        # Standard spread bet
        # picked_team_score + spread > opponent_score => win
        if picked_home:
            margin = home_score + spread - away_score
        else:
            margin = away_score + spread - home_score

        if margin > 0:
            return "win"
        elif margin == 0:
            return "void"
        else:
            return "loss"


def _team_matches(short_name, full_name):
    """Check if a short team name matches a full ESPN team name."""
    short_lower = short_name.lower()
    full_lower = full_name.lower()

    if short_lower == full_lower:
        return True
    if short_lower in full_lower or full_lower in short_lower:
        return True

    # Handle common abbreviations
    close = get_close_matches(short_lower, [full_lower], n=1, cutoff=0.6)
    return len(close) > 0


def calculate_payout(odds_str, wager, result, platform=None, stored_payout=None):
    """
    Calculate payout and profit for a bet.

    odds_str: American odds like "-110", "+150", or "n/a"
    wager: float wager amount
    result: "win", "loss", or "void"
    platform: sportsbook/platform name
    stored_payout: optional payout value already captured at log time

    Returns: (payout, profit)
    """
    wager = float(wager)

    if result == "void":
        return round(wager, 2), 0.00

    if result == "loss":
        return 0.00, round(-wager, 2)

    # result == "win"
    odds_str = str(odds_str).strip()
    platform_name = str(platform or "").strip().lower()

    if odds_str.lower() in ("n/a", "nan", ""):
        if platform_name == "kalshi":
            # For Kalshi, use the exact max payout if it was captured at log time
            # (e.g. from a Kalshi share URL). Otherwise, keep payout/profit unknown.
            try:
                payout_val = float(stored_payout)
                if payout_val != payout_val or payout_val <= 0:  # NaN or non-positive
                    raise ValueError
                payout = round(payout_val, 2)
                profit = round(payout - wager, 2)
                return payout, profit
            except (TypeError, ValueError):
                logger.warning(
                    "Kalshi payout unknown (missing odds and stored payout) for wager=%s",
                    wager,
                )
                return 0.00, 0.00

        logger.warning(f"Cannot calculate payout: odds are '{odds_str}' for wager={wager}")
        return 0.00, 0.00

    try:
        odds = int(float(odds_str))
    except (ValueError, TypeError):
        logger.warning(f"Cannot parse odds '{odds_str}' for wager={wager}")
        return 0.00, 0.00

    if odds < 0:
        # Favorite: profit = wager * 100 / abs(odds)
        profit = wager * 100.0 / abs(odds)
    else:
        # Underdog: profit = wager * odds / 100
        profit = wager * odds / 100.0

    payout = round(wager + profit, 2)
    profit = round(profit, 2)
    return payout, profit


def _format_settlement_detail(line_str, game_str, result, payout, profit, platform=""):
    """Build a user-facing settlement detail line."""
    icon = {"win": "W", "loss": "L", "void": "P"}[result]
    msg = (
        f"  [{icon}] {line_str} ({game_str}) -> {result}, "
        f"payout={payout:.2f}, profit={profit:+.2f}"
    )

    # Explicitly flag unknown Kalshi win payouts for manual correction.
    if (
        str(platform or "").strip().lower() == "kalshi"
        and result == "win"
        and float(payout) == 0.0
    ):
        msg += " (payout unknown -- manual review needed)"

    return msg


def settle_pending_bets(csv_path=None, league="mens"):
    """
    Settle all pending bets in betting_history.csv.

    Loads the CSV, finds pending bets, groups by date, fetches ESPN scores,
    determines results, and updates the CSV.

    Returns a summary dict:
        {settled: int, still_pending: int, details: list[str]}
    """
    if csv_path is None:
        csv_path = BETTING_HISTORY

    if not os.path.exists(csv_path):
        return {"settled": 0, "still_pending": 0, "details": ["No betting history file found."]}

    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"Could not read CSV {csv_path}: {e}")
        return {"settled": 0, "still_pending": 0, "details": [f"CSV read error: {e}"]}

    pending = df[df["result"] == "pending"]

    if len(pending) == 0:
        return {"settled": 0, "still_pending": 0, "details": ["No pending bets."]}

    # Group by date to minimize ESPN API calls
    # Also fetch adjacent dates (day before/after) to handle bets logged on a
    # different day than the game was played.
    pending_dates = pending["date"].unique()
    games_by_date = {}

    eastern = pytz.timezone("US/Eastern")

    fetched_dates = set()
    for date_str in pending_dates:
        try:
            date_obj = datetime.strptime(str(date_str), "%Y-%m-%d")
            date_obj = eastern.localize(date_obj)
        except ValueError:
            logger.warning(f"Could not parse date '{date_str}', skipping")
            continue
        for offset in [timedelta(0), timedelta(days=-1), timedelta(days=1)]:
            d = date_obj + offset
            d_str = d.strftime("%Y-%m-%d")
            if d_str not in fetched_dates:
                fetched_dates.add(d_str)
                try:
                    games_by_date[d_str] = fetch_completed_games(d, league=league)
                except Exception as e:
                    logger.error(f"ESPN API error for {d_str}: {e}")
                    games_by_date[d_str] = {}

    settled_count = 0
    still_pending = 0
    details = []

    for idx in pending.index:
        try:
            row = df.loc[idx]
            date_str = row["date"]
            game_str = row["game"]
            line_str = row["line"]
            odds_str = row["odds"]
            wager = row["wager"]

            # Collect completed games from the bet date and adjacent dates
            date_obj = datetime.strptime(str(date_str), "%Y-%m-%d")
            date_obj = eastern.localize(date_obj)
            completed = {}
            for offset in [timedelta(0), timedelta(days=-1), timedelta(days=1)]:
                d_str = (date_obj + offset).strftime("%Y-%m-%d")
                completed.update(games_by_date.get(d_str, {}))

            if not completed:
                still_pending += 1
                details.append(f"  No games found for {date_str}: {line_str}")
                continue

            # Match bet to game (try game field first, then fall back to team from line)
            game_result = match_bet_to_game(game_str, completed)
            if game_result is None:
                game_result = match_bet_by_team(line_str, completed)
            if game_result is None:
                still_pending += 1
                details.append(f"  No match for {game_str}: {line_str}")
                continue

            # Parse the bet line
            parsed = parse_bet_line(line_str)
            if parsed is None:
                still_pending += 1
                details.append(f"  Could not parse line: {line_str}")
                continue

            # Determine result
            result = determine_bet_result(parsed, game_result, game_str)
            if result is None:
                still_pending += 1
                details.append(f"  Could not determine result: {line_str}")
                continue

            # Calculate payout
            payout, profit = calculate_payout(
                odds_str,
                wager,
                result,
                platform=row.get("platform", ""),
                stored_payout=row.get("payout", ""),
            )

            # Update the row
            df.at[idx, "result"] = result
            df.at[idx, "payout"] = payout
            df.at[idx, "profit"] = profit

            settled_count += 1
            details.append(
                _format_settlement_detail(
                    line_str,
                    game_str,
                    result,
                    payout,
                    profit,
                    platform=row.get("platform", ""),
                )
            )
        except Exception as e:
            logger.error(f"Error settling bet at index {idx}: {e}")
            still_pending += 1
            details.append(f"  Error processing bet: {e}")

    # Save updated CSV (always write back work done so far)
    try:
        df.to_csv(csv_path, index=False)
    except (IOError, PermissionError) as e:
        logger.error(f"Could not write CSV {csv_path}: {e}")
        details.append(f"WARNING: Could not save results to CSV: {e}")

    return {
        "settled": settled_count,
        "still_pending": still_pending,
        "details": details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settle pending bets from betting_history.csv.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League scoreboard to use: mens or womens (aliases supported).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SETTLING PENDING BETS")
    print("=" * 60)

    summary = settle_pending_bets(league=args.league)

    print(f"\nSettled: {summary['settled']}")
    print(f"Still pending: {summary['still_pending']}")

    if summary["details"]:
        print("\nDetails:")
        for d in summary["details"]:
            print(d)
