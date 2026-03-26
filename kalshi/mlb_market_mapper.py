"""Map MLB games to Kalshi market tickers.

MLB Kalshi ticker format (confirmed from live markets):
    KXMLBGAME-{YYMMMDDHHMMAWAY}{HOME}-{YES_ABBR}
    e.g. KXMLBGAME-26MAR281610BOSCIN-CIN

    KXMLBSPREAD-{YYMMMDDHHMMAWAY}{HOME}-{ABBR}{SPREAD}
    e.g. KXMLBSPREAD-26MAR271915KCATL-KC2  (KC wins by over 1.5)

    KXMLBTOTAL-{YYMMMDDHHMMAWAY}{HOME}-{THRESHOLD}
    e.g. KXMLBTOTAL-26MAR271915KCATL-9  (over 9 total runs)
"""

from datetime import datetime
from typing import Optional


# Standard MLB team abbreviations used in Kalshi tickers
MLB_ABBREVIATIONS = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# Reverse lookup
ABBR_TO_TEAM = {v: k for k, v in MLB_ABBREVIATIONS.items()}


class MLBMarketMapper:
    """Maps MLB games to Kalshi market tickers."""

    def __init__(self, markets: list):
        """Initialize with a list of Kalshi market dicts."""
        self.markets = markets
        self._index = {}
        for m in markets:
            ticker = m.get("ticker", "")
            self._index[ticker] = m

    @staticmethod
    def _parse_ticker_teams(ticker: str) -> tuple:
        """Extract (away_abbr, home_abbr) from a Kalshi MLB ticker.

        Ticker format: KXMLB{TYPE}-{YYMMMDDHHMMAWAYABBRHOMEABBR}-{YES_ABBR}
        e.g. KXMLBGAME-26MAR282040DETSD-SD -> (DET, SD)

        The teams portion starts after the 4-digit time (positions 9+) in the
        middle segment. We match against known abbreviations to split correctly.
        """
        parts = ticker.split("-")
        if len(parts) < 2:
            return ("", "")
        middle = parts[1]
        # Skip date (YYMMMDD = 7 chars) + time (HHMM = 4 chars) = 11 chars
        teams_str = middle[11:] if len(middle) > 11 else ""
        if not teams_str:
            return ("", "")

        # Try all known abbreviations to find the split point
        all_abbrs = sorted(MLB_ABBREVIATIONS.values(), key=len, reverse=True)
        for away in all_abbrs:
            if teams_str.startswith(away):
                home = teams_str[len(away):]
                if home in MLB_ABBREVIATIONS.values():
                    return (away, home)
        return ("", "")

    def find_market(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        market_type: str = "GAME",
    ) -> Optional[dict]:
        """Find a Kalshi market matching an MLB game.

        Args:
            home_team: Full team name (e.g., "New York Yankees")
            away_team: Full team name
            game_date: Game date
            market_type: "GAME", "SPREAD", or "TOTAL"

        Returns:
            Market dict if found, None otherwise.
        """
        home_abbr = MLB_ABBREVIATIONS.get(home_team, "")
        away_abbr = MLB_ABBREVIATIONS.get(away_team, "")

        if not home_abbr or not away_abbr:
            return None

        date_str = game_date.strftime("%y%b%d").upper()  # 26MAR28 format
        prefix = f"KXMLB{market_type}"

        for ticker, market in self._index.items():
            if not ticker.startswith(prefix):
                continue
            # Parse teams from ticker structure (not substring matching)
            t_away, t_home = self._parse_ticker_teams(ticker)
            if t_away == away_abbr and t_home == home_abbr:
                # Verify date matches
                if date_str in ticker.upper():
                    return market

        # Fallback: match on title text
        for ticker, market in self._index.items():
            if not ticker.startswith(prefix):
                continue
            title = market.get("title", "").lower()
            home_check = home_team.lower() in title or home_abbr.lower() in title
            away_check = away_team.lower() in title or away_abbr.lower() in title
            if home_check and away_check:
                return market

        return None

    def get_yes_team(self, ticker: str) -> Optional[str]:
        """Determine which team is YES from the ticker suffix."""
        # Ticker suffix after the last dash is typically the YES team abbreviation
        parts = ticker.rsplit("-", 1)
        if len(parts) == 2:
            suffix = parts[1].upper()
            return ABBR_TO_TEAM.get(suffix)
        return None
