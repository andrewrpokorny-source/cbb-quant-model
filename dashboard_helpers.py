"""Helpers for the Streamlit dashboard that can be imported without Streamlit."""

import re
from datetime import datetime, timedelta

# MLB abbreviations for ticker parsing (sorted longest-first for greedy match)
_MLB_ABBRS = sorted([
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
], key=len, reverse=True)


def extract_ticker_teams(ticker: str) -> tuple[str, str]:
    """Extract (away_abbr, home_abbr) from a Kalshi ticker.

    MLB tickers encode both teams: KXMLBGAME-{YYMMMDDHHMMAWAYHHOME}-{YES}
    CBB tickers only have the YES team as the suffix.

    Returns ("", "") on parse failure.
    """
    parts = ticker.split("-")
    if len(parts) < 3:
        return ("", "")

    middle = parts[1]

    # MLB tickers: date is YYMMMDD (7 chars), time is HHMM (4 chars), then teams
    if "KXMLB" in parts[0].upper():
        if len(middle) < 12:
            return ("", "")
        teams_str = middle[11:]
        for away in _MLB_ABBRS:
            if teams_str.startswith(away):
                home = teams_str[len(away):]
                if home in _MLB_ABBRS:
                    return (away, home)
        return ("", "")

    # CBB/other: can only extract the YES team from the suffix
    yes_abbr = re.match(r"([A-Z]+)", parts[-1].upper())
    if yes_abbr:
        return (yes_abbr.group(1), "")
    return ("", "")


def position_matches_game(ticker: str, game: dict, ticker_league: str) -> bool:
    """Check if a Kalshi position ticker matches a specific live game.

    For MLB, verifies both teams in the ticker match both teams in the game.
    For CBB, falls back to YES-team-only matching (ticker doesn't encode both).
    """
    if game.get("league") != ticker_league:
        return False

    game_teams = {game.get("home_abbr", ""), game.get("away_abbr", "")}
    game_teams.discard("")

    away, home = extract_ticker_teams(ticker)

    # MLB: both teams must match
    if away and home:
        ticker_teams = {away, home}
        return ticker_teams == game_teams

    # CBB fallback: at least the YES team (suffix) must be in the game
    parts = ticker.split("-")
    if len(parts) >= 3:
        yes_abbr_match = re.match(r"([A-Z]+)", parts[-1].upper())
        if yes_abbr_match:
            return yes_abbr_match.group(1) in game_teams

    return False


def filter_recent_kalshi(bets, *, cutoff_days=7, limit=8):
    """Return recent settled Kalshi results, sorted newest-first.

    Args:
        bets: list of dicts (as read by csv.DictReader from betting_history.csv)
        cutoff_days: only include results from the last N days
        limit: max number of results to return
    """
    cutoff = (datetime.now() - timedelta(days=cutoff_days)).strftime("%Y-%m-%d")
    filtered = [
        r for r in bets
        if (r.get("platform") or "").strip().upper() == "KALSHI"
        and (r.get("date") or "") >= cutoff
        and (r.get("result") or "").strip().lower() in ("win", "loss", "void")
    ]
    filtered.sort(key=lambda r: r.get("date") or "", reverse=True)
    return filtered[:limit]
