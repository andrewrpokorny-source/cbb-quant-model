"""Map MLB games to Kalshi market tickers.

MLB Kalshi tickers are expected to follow the pattern:
    KXNMLBGAME-{DATE}{TEAMS}-{SUFFIX}
where SUFFIX is the YES team abbreviation.

This mapper is much simpler than CBB because there are only 30 MLB teams
with stable, standardized abbreviations.
"""

import re
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

        date_str = game_date.strftime("%d%b%y").upper()
        prefix = f"KXNMLB{market_type}"

        for ticker, market in self._index.items():
            if not ticker.startswith(prefix):
                continue
            ticker_upper = ticker.upper()
            # Check if both team abbreviations appear in the ticker
            if home_abbr in ticker_upper and away_abbr in ticker_upper:
                # Check date if encoded in ticker
                if date_str in ticker_upper or not date_str:
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
