"""Map ESPN games to Kalshi market tickers."""

import re
from difflib import SequenceMatcher
from typing import Optional, List
from datetime import datetime


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching - extract just the school name."""
    # Strip gendered prefix+mascot first so suffix removal doesn't leave orphan "Lady"
    # (NCAAW: "Lady Rebels", "Lady Golden Eagles", etc.)
    name = re.sub(r"\s+Lady\s+[\w\s]+$", "", name, flags=re.IGNORECASE)

    # Remove common suffixes/mascots
    suffixes = [
        "Wildcats", "Bulldogs", "Tigers", "Bears", "Jayhawks", "Blue Devils",
        "Tar Heels", "Volunteers", "Boilermakers", "Cougars", "Cardinals",
        "Gators", "Wolverines", "Hoosiers", "Badgers", "Fighting Illini",
        "Hawkeyes", "Orange", "Razorbacks", "Cavaliers", "Golden Eagles",
        "Bluejays", "Musketeers", "Red Storm", "Friars", "Pirates", "Ducks",
        "Trojans", "Buffaloes", "Bearcats", "Aztecs", "Crimson Tide",
        "Spartans", "Buckeyes", "Huskies", "Bruins", "Lobos", "Catamounts",
        "Paladins", "Toreros", "Lancers", "Trailblazers", "Mavericks",
        "Coyotes", "Owls", "Aggies", "Longhorns", "Red Raiders", "Sooners",
        "Cowboys", "Cowgirls", "Mountaineers", "Cyclones", "Horned Frogs",
        "Dons", "Gaels", "Broncos", "Pilots", "Waves", "Lions", "Waves",
        "Panthers", "Zags", "Demon Deacons", "Yellow Jackets", "Seminoles",
        "Hurricanes", "Hokies", "Wolfpack", "Commodores", "Rebels",
        "Gamecocks", "Ladyjacks", "Devilettes", "Lopes", "Mean Green",
        "Green Wave", "Roadrunners", "Bulls", "Shockers", "Thunderbirds",
        "Runnin' Bulldogs", "Mustangs", "Bobcats", "Penguins", "Flyers",
        "Billikens", "Rams", "Bison", "Terriers", "Retrievers", "Peacocks",
        "Anteaters", "49ers", "Highlanders", "Matadors", "Titans",
        "Vaqueros", "Islanders", "Salukis",
    ]

    for suffix in suffixes:
        name = re.sub(rf"\s+{suffix}$", "", name, flags=re.IGNORECASE)

    return name.strip()


def extract_school_keyword(name: str) -> str:
    """Extract the key school name for fuzzy matching."""
    normalized = normalize_team_name(name)

    # Handle special cases - order matters (more specific first)
    special = [
        ("South Carolina Upstate", "usc upstate"),
        ("USC Upstate", "usc upstate"),
        ("UNC Asheville", "unc asheville"),
        ("North Carolina", "north carolina"),
        ("South Carolina", "south carolina"),
        ("Michigan State", "michigan state"),
        ("Michigan St.", "michigan state"),
        ("Ohio State", "ohio state"),
        ("Ohio St.", "ohio state"),
        ("San Diego State", "san diego state"),
        ("San Diego St.", "san diego state"),
        ("Fresno State", "fresno"),
        ("Washington State", "washington state"),
        ("New Mexico State", "new mexico state"),
        ("New Mexico St.", "new mexico state"),
        ("St. John's", "st. john"),
        ("Saint John's", "st. john"),
    ]

    for key, val in special:
        if key.lower() in normalized.lower():
            return val

    return normalized.lower()


class MarketMapper:
    """Map games to Kalshi market tickers."""

    def __init__(self, kalshi_markets: list):
        """
        Initialize with list of available Kalshi markets.

        Args:
            kalshi_markets: List of market dicts from Kalshi API
        """
        self.markets = kalshi_markets
        self._build_index()

    def _build_index(self):
        """Build search indices from markets."""
        self.by_ticker = {m.get("ticker", ""): m for m in self.markets}
        self.by_event = {}  # event_ticker -> list of markets

        for m in self.markets:
            event = m.get("event_ticker", "")
            if event:
                if event not in self.by_event:
                    self.by_event[event] = []
                self.by_event[event].append(m)

    def _parse_kalshi_date(self, ticker: str) -> Optional[datetime]:
        """Parse date from Kalshi ticker like KXNCAAMBSPREAD-26JAN21XAVCREI.

        Format is YYMMMDD: 26JAN21 = 2026-01-21
        """
        match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ticker)
        if match:
            year_short, month_str, day = match.groups()
            months = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
            }
            month = months.get(month_str, 1)
            year_full = 2000 + int(year_short)
            try:
                return datetime(year_full, month, int(day))
            except ValueError:
                pass
        return None

    def _teams_in_rules(self, rules: str, team_keyword: str) -> bool:
        """Check if team appears in a text field (e.g. rules_primary).

        Handles abbreviation mismatches (e.g. 'State' vs 'St.') by trying
        expanded and abbreviated variants.
        """
        keyword = team_keyword.lower()
        text = rules.lower()

        if keyword in text:
            return True

        # Try abbreviation variants: "state" -> "st.", "saint" -> "st.",
        # "st." -> "state", "st." -> "saint"
        variants = [
            (r"\bstate\b", "st."),
            (r"\bst\.", "state"),
            (r"\bsaint\b", "st."),
            (r"\bst\.", "saint"),
        ]
        for pattern, replacement in variants:
            alt = re.sub(pattern, replacement, keyword)
            if alt != keyword and alt in text:
                return True

        return False

    def _similarity(self, a: str, b: str) -> float:
        """Calculate string similarity ratio."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def find_market(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        spread: float,
    ) -> Optional[dict]:
        """
        Find Kalshi spread market matching a game.

        Kalshi ticker format: KXNCAAMBSPREAD-26JAN21XAVCREI-XAV9
        - 26JAN21 = date
        - XAVCREI = away team + home team abbreviations
        - XAV9 = team covering spread + spread value

        Args:
            home_team: Home team name (ESPN format)
            away_team: Away team name (ESPN format)
            game_date: Game date
            spread: Vegas spread (home team perspective, negative = home favored)

        Returns:
            Market dict if found, None otherwise
        """
        home_keyword = extract_school_keyword(home_team)
        away_keyword = extract_school_keyword(away_team)

        # Find spread markets for this date
        spread_markets = [
            m for m in self.markets
            if "SPREAD" in m.get("ticker", "")
        ]

        best_match = None
        best_score = 0

        for market in spread_markets:
            ticker = market.get("ticker", "")
            rules = market.get("rules_primary", "")
            title = market.get("title", "")

            # Check date match
            market_date = self._parse_kalshi_date(ticker)
            if not market_date:
                continue

            # Allow 1 day tolerance for timezone differences
            date_diff = abs((market_date.date() - game_date.date()).days)
            if date_diff > 1:
                continue

            # Check if BOTH teams in rules (required)
            home_in_rules = self._teams_in_rules(rules, home_keyword)
            away_in_rules = self._teams_in_rules(rules, away_keyword)

            if not (home_in_rules and away_in_rules):
                continue

            # Found a matching game - now find the right spread
            floor_strike = market.get("floor_strike", 0)

            # Score based on spread match
            # ESPN spread is from home perspective (negative = home favored)
            # Kalshi has separate markets for each team covering
            spread_diff = abs(floor_strike - abs(spread))

            # Prefer exact spread match
            if spread_diff < 0.5:
                score = 100
            elif spread_diff < 1.5:
                score = 80
            elif spread_diff < 3:
                score = 60
            else:
                score = 40

            # Bonus for team name match in title
            if home_keyword in title.lower() or away_keyword in title.lower():
                score += 10

            if score > best_score:
                best_score = score
                best_match = market

        return best_match

    def find_all_markets_for_game(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
    ) -> List[dict]:
        """Find all Kalshi markets for a game (useful for debugging)."""
        home_keyword = extract_school_keyword(home_team)
        away_keyword = extract_school_keyword(away_team)

        matches = []
        for market in self.markets:
            rules = market.get("rules_primary", "")
            ticker = market.get("ticker", "")

            market_date = self._parse_kalshi_date(ticker)
            if not market_date:
                continue

            date_diff = abs((market_date.date() - game_date.date()).days)
            if date_diff > 1:
                continue

            if self._teams_in_rules(rules, home_keyword) and self._teams_in_rules(rules, away_keyword):
                matches.append(market)

        return matches

    def get_market_prices(self, market: dict) -> dict:
        """
        Extract prices from market dict.

        Args:
            market: Market dict from Kalshi API

        Returns:
            Dict with yes_price, no_price (0-100 scale)
        """
        # yes_ask is what you pay to buy YES (the actual cost to enter)
        # This is what matters for edge calculation -- using bid overstates edge
        yes_ask = market.get("yes_ask")
        yes_price = yes_ask if yes_ask is not None else market.get("last_price")
        no_ask = market.get("no_ask")
        no_price = no_ask if no_ask is not None else (100 - yes_price if yes_price is not None else None)

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "ticker": market.get("ticker", ""),
            "title": market.get("title", ""),
            "floor_strike": market.get("floor_strike", 0),
        }
