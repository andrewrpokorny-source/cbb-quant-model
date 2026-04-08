"""Helpers for the Streamlit dashboard that can be imported without Streamlit."""

import re
from datetime import datetime, timedelta, timezone

# MLB abbreviations for ticker parsing (sorted longest-first for greedy match)
_MLB_ABBRS = sorted([
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SD", "SF", "SEA", "STL", "TB", "TEX", "TOR", "WSH",
], key=len, reverse=True)


_MONTH_MAP = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def extract_ticker_date(ticker: str) -> str:
    """Extract game date from a Kalshi ticker as 'YYYY-MM-DD'.

    Ticker middle segment starts with YYMMMDD (e.g. '26APR06' = 2026-04-06).
    Returns '' on parse failure.
    """
    parts = ticker.split("-")
    if len(parts) < 3:
        return ""
    middle = parts[1]
    if len(middle) < 7:
        return ""
    yy = middle[:2]
    mmm = middle[2:5].upper()
    dd = middle[5:7]
    mm = _MONTH_MAP.get(mmm)
    if not mm or not yy.isdigit() or not dd.isdigit():
        return ""
    return f"20{yy}-{mm}-{dd}"


_ET_OFFSET_STD = timedelta(hours=-5)   # EST
_ET_OFFSET_DST = timedelta(hours=-4)   # EDT


def utc_date_to_eastern(utc_str: str) -> str:
    """Convert a UTC ISO date string to an Eastern Time date string (YYYY-MM-DD).

    Ticker dates are in ET, so ESPN UTC dates must be converted to ET before
    comparison.  Returns '' on parse failure.  Handles both 'YYYY-MM-DDTHH:MMZ'
    and plain 'YYYY-MM-DD' inputs.
    """
    if not utc_str or len(utc_str) < 10:
        return ""
    # Plain date (no time component) -- pass through if valid
    if "T" not in utc_str:
        candidate = utc_str[:10]
        if len(candidate) == 10 and candidate[4] == "-" and candidate[7] == "-":
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
                return candidate
            except ValueError:
                return ""
        return ""
    try:
        clean = utc_str.rstrip("Z")
        dt = datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
        # Approximate DST: EDT (UTC-4) Mar second Sun - Nov first Sun,
        # EST (UTC-5) otherwise.  Use the standard US rule.
        et = dt + _ET_OFFSET_STD
        # Check if DST applies (second Sunday of March to first Sunday of Nov)
        year = et.year
        # March: second Sunday
        mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
        # November: first Sunday
        nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
        dst_start = dst_start.replace(hour=7)  # 2 AM ET = 7 AM UTC
        dst_end = dst_end.replace(hour=6)      # 2 AM EDT = 6 AM UTC
        if dst_start <= dt < dst_end:
            et = dt + _ET_OFFSET_DST
        return et.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


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
    When the game dict includes a 'game_date' key, the ticker date must also match
    to prevent stale positions from prior days ghosting onto today's games.
    """
    if game.get("league") != ticker_league:
        return False

    # Date check: if both ticker and game have dates, they must match
    game_date = game.get("game_date", "")
    if game_date:
        ticker_date = extract_ticker_date(ticker)
        if ticker_date and ticker_date != game_date:
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
