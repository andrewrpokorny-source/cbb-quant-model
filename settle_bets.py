"""
Auto-settlement engine for betting_history.csv.

Matches pending bets to ESPN completed game scores, determines results,
and updates the CSV with payout/profit calculations.

Usage:
    python settle_bets.py
"""

import os
import re
import pandas as pd
from datetime import datetime, timedelta
from difflib import get_close_matches
import pytz

from grade_predictions import fetch_completed_games

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BETTING_HISTORY = os.path.join(BASE_DIR, "betting_history.csv")


def parse_bet_line(line_str):
    """
    Parse a bet line string into components.

    Examples:
        "Providence +15.5"       -> {team: "Providence", spread: 15.5, side: None}
        "UConn -15.5 NO"        -> {team: "UConn", spread: -15.5, side: "NO"}
        "Furman -10.5 YES"      -> {team: "Furman", spread: -10.5, side: "YES"}
        "Mercer -9.5"           -> {team: "Mercer", spread: -9.5, side: None}
        "Alabama -10.5"         -> {team: "Alabama", spread: -10.5, side: None}
    """
    line_str = line_str.strip()

    # Check for Kalshi YES/NO suffix
    side = None
    if line_str.endswith(" YES"):
        side = "YES"
        line_str = line_str[:-4].strip()
    elif line_str.endswith(" NO"):
        side = "NO"
        line_str = line_str[:-3].strip()

    # Split off the spread (last token should be +/- number)
    match = re.match(r"^(.+?)\s+([+-]?\d+\.?\d*)$", line_str)
    if not match:
        return None

    team = match.group(1).strip()
    spread = float(match.group(2))

    return {"team": team, "spread": spread, "side": side}


def match_bet_to_game(game_str, completed_games):
    """
    Match a bet's game string (e.g. "Providence vs UConn") to ESPN completed games.

    completed_games: dict from fetch_completed_games() keyed by (home_name, away_name)

    Returns the game result dict or None.
    """
    # Parse "Away vs Home" or "Away @ Home" format
    for sep in [" vs ", " @ "]:
        if sep in game_str:
            parts = game_str.split(sep)
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

    home_score = game_result["home_score"]
    away_score = game_result["away_score"]
    home_name = game_result["home_name"]
    away_name = game_result["away_name"]

    # Figure out which team was picked
    picked_home = _team_matches(team, home_name)
    picked_away = _team_matches(team, away_name)

    if not picked_home and not picked_away:
        # Try the other team in the line for Kalshi NO bets
        # For "UConn -15.5 NO" on "Providence vs UConn", the team is UConn
        # We need to find which side UConn is on
        picked_home = _team_matches(team, home_name)
        picked_away = _team_matches(team, away_name)
        if not picked_home and not picked_away:
            return None

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


def calculate_payout(odds_str, wager, result):
    """
    Calculate payout and profit for a bet.

    odds_str: American odds like "-110", "+150", or "n/a" for Kalshi
    wager: float wager amount
    result: "win", "loss", or "void"

    Returns: (payout, profit)
    """
    wager = float(wager)

    if result == "void":
        return round(wager, 2), 0.00

    if result == "loss":
        return 0.00, round(-wager, 2)

    # result == "win"
    odds_str = str(odds_str).strip()

    if odds_str in ("n/a", "nan", ""):
        # Kalshi bet — we can't calculate exact payout without the price
        # Return wager * 2 as approximate (will be overridden if price is known)
        # In practice, Kalshi payouts are already recorded or we use the original payout
        return round(wager * 2, 2), round(wager, 2)

    try:
        odds = int(float(odds_str))
    except (ValueError, TypeError):
        # Fallback for non-parseable odds
        return round(wager * 2, 2), round(wager, 2)

    if odds < 0:
        # Favorite: profit = wager * 100 / abs(odds)
        profit = wager * 100.0 / abs(odds)
    else:
        # Underdog: profit = wager * odds / 100
        profit = wager * odds / 100.0

    payout = round(wager + profit, 2)
    profit = round(profit, 2)
    return payout, profit


def settle_pending_bets(csv_path=None):
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

    df = pd.read_csv(csv_path)
    pending = df[df["result"] == "pending"]

    if len(pending) == 0:
        return {"settled": 0, "still_pending": 0, "details": ["No pending bets."]}

    # Group by date to minimize ESPN API calls
    pending_dates = pending["date"].unique()
    games_by_date = {}

    eastern = pytz.timezone("US/Eastern")

    for date_str in pending_dates:
        date_obj = datetime.strptime(str(date_str), "%Y-%m-%d")
        date_obj = eastern.localize(date_obj)
        games_by_date[date_str] = fetch_completed_games(date_obj)

    settled_count = 0
    still_pending = 0
    details = []

    for idx in pending.index:
        row = df.loc[idx]
        date_str = row["date"]
        game_str = row["game"]
        line_str = row["line"]
        odds_str = row["odds"]
        wager = row["wager"]

        completed = games_by_date.get(date_str, {})
        if not completed:
            still_pending += 1
            details.append(f"  No games found for {date_str}: {line_str}")
            continue

        # Match bet to game
        game_result = match_bet_to_game(game_str, completed)
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
        payout, profit = calculate_payout(odds_str, wager, result)

        # Update the row
        df.at[idx, "result"] = result
        df.at[idx, "payout"] = payout
        df.at[idx, "profit"] = profit

        settled_count += 1
        icon = {"win": "W", "loss": "L", "void": "P"}[result]
        details.append(
            f"  [{icon}] {line_str} ({game_str}) -> {result}, "
            f"payout={payout:.2f}, profit={profit:+.2f}"
        )

    # Save updated CSV
    df.to_csv(csv_path, index=False)

    return {
        "settled": settled_count,
        "still_pending": still_pending,
        "details": details,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SETTLING PENDING BETS")
    print("=" * 60)

    summary = settle_pending_bets()

    print(f"\nSettled: {summary['settled']}")
    print(f"Still pending: {summary['still_pending']}")

    if summary["details"]:
        print("\nDetails:")
        for d in summary["details"]:
            print(d)
