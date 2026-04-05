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
    def _parse_ticker_middle(ticker: str) -> dict:
        """Parse the middle segment of a Kalshi MLB ticker.

        Ticker format: KXMLB{TYPE}-{YYMMMDDHHMMAWAYABBRHOMEABBR}-{YES_ABBR}
        e.g. KXMLBGAME-26MAR282040DETSD-SD

        Returns dict with keys: away, home, hhmm (all strings, empty on failure).
        """
        parts = ticker.split("-")
        if len(parts) < 2:
            return {"away": "", "home": "", "hhmm": ""}
        middle = parts[1]
        # Date = YYMMMDD (7 chars), time = HHMM (4 chars), teams = rest
        if len(middle) < 12:
            return {"away": "", "home": "", "hhmm": ""}

        hhmm = middle[7:11]
        teams_str = middle[11:]
        if not teams_str:
            return {"away": "", "home": "", "hhmm": hhmm}

        all_abbrs = sorted(MLB_ABBREVIATIONS.values(), key=len, reverse=True)
        for away in all_abbrs:
            if teams_str.startswith(away):
                home = teams_str[len(away):]
                if home in MLB_ABBREVIATIONS.values():
                    return {"away": away, "home": home, "hhmm": hhmm}
        return {"away": "", "home": "", "hhmm": hhmm}

    @classmethod
    def _parse_ticker_teams(cls, ticker: str) -> tuple:
        """Extract (away_abbr, home_abbr) from a Kalshi MLB ticker."""
        parsed = cls._parse_ticker_middle(ticker)
        return (parsed["away"], parsed["home"])

    def find_market(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        market_type: str = "GAME",
        game_time: str = "",
    ) -> Optional[dict]:
        """Find a Kalshi market matching an MLB game.

        Args:
            home_team: Full team name (e.g., "New York Yankees")
            away_team: Full team name
            game_date: Game datetime (date used for matching, time for doubleheaders)
            market_type: "GAME", "SPREAD", or "TOTAL"
            game_time: Optional "HH:MM" UTC time to disambiguate doubleheaders

        Returns:
            Market dict if found, None otherwise.
        """
        home_abbr = MLB_ABBREVIATIONS.get(home_team, "")
        away_abbr = MLB_ABBREVIATIONS.get(away_team, "")

        if not home_abbr or not away_abbr:
            return None

        date_str = game_date.strftime("%y%b%d").upper()  # 26MAR28 format
        # Normalize game_time to HHMM for ticker matching
        time_hhmm = game_time.replace(":", "")[:4] if game_time else ""
        prefix = f"KXMLB{market_type}"

        candidates = []
        for ticker, market in self._index.items():
            if not ticker.startswith(prefix):
                continue
            parsed = self._parse_ticker_middle(ticker)
            if parsed["away"] == away_abbr and parsed["home"] == home_abbr:
                if date_str in ticker.upper():
                    candidates.append((ticker, market, parsed["hhmm"]))

        if not candidates:
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

        # Single match -- return it
        if len(candidates) == 1:
            return candidates[0][1]

        # Multiple matches (doubleheader) -- use game_time to disambiguate
        if time_hhmm:
            for ticker, market, ticker_hhmm in candidates:
                if ticker_hhmm == time_hhmm:
                    return market
            # No exact time match -- pick closest
            best = min(candidates, key=lambda c: abs(int(c[2] or "0") - int(time_hhmm or "0")))
            return best[1]

        # No time provided -- return first match
        return candidates[0][1]

    def get_yes_team(self, ticker: str) -> Optional[str]:
        """Determine which team is YES from the ticker suffix."""
        # Ticker suffix after the last dash is typically the YES team abbreviation
        parts = ticker.rsplit("-", 1)
        if len(parts) == 2:
            suffix = parts[1].upper()
            return ABBR_TO_TEAM.get(suffix)
        return None
