"""Map ESPN games to Polymarket sports market tokens.

Polymarket sports markets are fetched via the CLI's ``sports list`` command.
Because the exact JSON schema may evolve, field names are auto-detected from
the first market dict with fallbacks for common variants.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from kalshi.market_mapper import normalize_team_name, extract_school_keyword


def _first_present(d: dict, keys: list[str]):
    """Return the value for the first key found in *d*, or ``None``."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


class PolymarketMarketMapper:
    """Map games to Polymarket sports market tokens."""

    # Candidate field names the CLI might use for each concept.
    _TITLE_CANDIDATES = ["question", "title", "name", "groupItemTitle", "market_title"]
    _DATE_CANDIDATES = ["game_start_time", "start_time", "game_date", "startDate",
                        "endDate", "end_date_iso", "game_start"]
    _TOKEN_CANDIDATES = ["clobTokenIds", "token_id", "condition_id", "conditionId",
                         "id", "market_id"]
    _OUTCOME_PRICE_CANDIDATES = ["outcomePrices", "outcome_prices"]
    _OUTCOMES_CANDIDATES = ["outcomes", "groupItemTitle"]
    _MARKET_TYPE_CANDIDATES = ["market_type", "marketType", "bet_type", "type"]
    _SPREAD_CANDIDATES = ["spread_value", "spreadValue", "line", "handicap",
                          "floor_strike", "point_spread"]

    def __init__(self, markets: list[dict]):
        self.markets = markets
        self._detect_fields()
        self._build_index()

    def _detect_fields(self):
        """Auto-detect which JSON field names are present."""
        sample = self.markets[0] if self.markets else {}
        all_keys = set(sample.keys())

        # If markets contain nested 'markets' arrays (event-level wrapping),
        # flatten one level.
        if "markets" in sample and isinstance(sample["markets"], list):
            flat = []
            for event in self.markets:
                for m in event.get("markets", []):
                    m["_event"] = {
                        k: v for k, v in event.items() if k != "markets"
                    }
                    flat.append(m)
            self.markets = flat
            sample = flat[0] if flat else {}
            all_keys = set(sample.keys())

        def _pick(candidates):
            for c in candidates:
                if c in all_keys:
                    return c
            return candidates[0]  # fallback to first candidate

        self._f_title = _pick(self._TITLE_CANDIDATES)
        self._f_date = _pick(self._DATE_CANDIDATES)
        self._f_token = _pick(self._TOKEN_CANDIDATES)
        self._f_prices = _pick(self._OUTCOME_PRICE_CANDIDATES)
        self._f_outcomes = _pick(self._OUTCOMES_CANDIDATES)
        self._f_type = _pick(self._MARKET_TYPE_CANDIDATES)
        self._f_spread = _pick(self._SPREAD_CANDIDATES)

    def _build_index(self):
        """Build lookup structures."""
        self.by_id = {}
        for m in self.markets:
            token = self._get_token_id(m)
            if token:
                self.by_id[token] = m

    # -- Field accessors --

    def _get_title(self, market: dict) -> str:
        return str(market.get(self._f_title, "") or "")

    def _get_date(self, market: dict) -> Optional[datetime]:
        raw = market.get(self._f_date)
        if raw is None:
            # Try event-level date
            event = market.get("_event", {})
            raw = _first_present(event, self._DATE_CANDIDATES)
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            from datetime import timezone
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        raw = str(raw)
        # Try ISO with explicit timezone first
        if raw.endswith("Z") or "+" in raw[10:]:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        return None

    def _get_token_id(self, market: dict) -> Optional[str]:
        val = market.get(self._f_token)
        if isinstance(val, list):
            return val[0] if val else None
        return str(val) if val else None

    def _get_yes_token(self, market: dict) -> Optional[str]:
        val = market.get(self._f_token)
        if isinstance(val, list):
            return val[0] if val else None
        return str(val) if val else None

    def _get_no_token(self, market: dict) -> Optional[str]:
        val = market.get(self._f_token)
        if isinstance(val, list) and len(val) > 1:
            return val[1]
        return None

    def _get_market_type(self, market: dict) -> str:
        raw = market.get(self._f_type)
        if raw:
            raw = str(raw).lower()
            if "spread" in raw:
                return "spread"
            if "total" in raw or "over" in raw:
                return "total"
            if "money" in raw or "game" in raw or "winner" in raw or "moneyline" in raw:
                return "game"
            return raw
        # Infer from title
        title = self._get_title(market).lower()
        if "spread" in title or "wins by" in title:
            return "spread"
        if "total" in title or "over" in title or "under" in title:
            return "total"
        return "game"

    def _get_spread_value(self, market: dict) -> Optional[float]:
        val = market.get(self._f_spread)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
        # Try parsing from title (e.g. "Duke -5.5")
        title = self._get_title(market)
        match = re.search(r"[+-]?\d+\.?\d*", title)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        return None

    def _get_prices(self, market: dict) -> tuple[Optional[float], Optional[float]]:
        """Extract (yes_price, no_price) in 0-100 cent scale."""
        raw = market.get(self._f_prices)
        if isinstance(raw, list) and len(raw) >= 2:
            try:
                yes = float(raw[0]) * 100.0
                no = float(raw[1]) * 100.0
                return round(yes, 2), round(no, 2)
            except (TypeError, ValueError):
                pass
        return None, None

    # -- Team matching --

    def _extract_teams_from_title(self, market: dict) -> tuple[str, str]:
        """Try to parse two team names from the market title.

        Common patterns:
          "Will <Team A> beat <Team B>?"
          "<Team A> vs <Team B>"
          "<Team A> vs. <Team B>"
          "<Team A> @ <Team B>"
        """
        title = self._get_title(market)
        patterns = [
            r"(?:will\s+)?(.+?)\s+(?:beat|defeat)\s+(.+?)[\s?]*$",
            r"(.+?)\s+(?:vs\.?|@)\s+(.+?)(?:\s*[-\(]|$)",
            r"(.+?)\s+(?:vs\.?|@)\s+(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        return "", ""

    def _teams_match(self, market: dict, home_keyword: str, away_keyword: str) -> bool:
        """Check if both team keywords appear in the market title."""
        title = self._get_title(market).lower()
        # Also check event-level title if available
        event_title = str(market.get("_event", {}).get("title", "")).lower()
        combined = f"{title} {event_title}"

        home_found = home_keyword in combined
        away_found = away_keyword in combined

        if home_found and away_found:
            return True

        # Try abbreviation variants
        variants = [
            (r"\bstate\b", "st."), (r"\bst\.", "state"),
            (r"\bsaint\b", "st."), (r"\bst\.", "saint"),
        ]
        for pattern, replacement in variants:
            home_alt = re.sub(pattern, replacement, home_keyword)
            away_alt = re.sub(pattern, replacement, away_keyword)
            if not home_found and home_alt != home_keyword and home_alt in combined:
                home_found = True
            if not away_found and away_alt != away_keyword and away_alt in combined:
                away_found = True

        return home_found and away_found

    # -- Public matching API --

    def _time_distance(self, market: dict, game_time: str) -> int:
        """Minutes between the market's start time and the provided game_time.

        Both ``game_time`` and the market datetime are normalised to UTC
        before comparison. ``game_time`` is expected as ``"HH:MM"`` in UTC
        (the format used by ESPN and passed through by predict.py).

        Returns a large value if the market has no parseable time.
        """
        if not game_time:
            return 0
        market_dt = self._get_date(market)
        if market_dt is None:
            return 9999
        try:
            parts = game_time.replace(":", "")[:4]
            gt_minutes = int(parts[:2]) * 60 + int(parts[2:4])

            # Normalize market time to UTC if it's timezone-aware
            if hasattr(market_dt, "utcoffset") and market_dt.utcoffset() is not None:
                from datetime import timezone
                market_utc = market_dt.astimezone(timezone.utc)
                mt_minutes = market_utc.hour * 60 + market_utc.minute
            else:
                # Assume naive datetimes are already UTC (Polymarket convention)
                mt_minutes = market_dt.hour * 60 + market_dt.minute

            return abs(gt_minutes - mt_minutes)
        except (ValueError, IndexError):
            return 9999

    def _pick_closest(self, candidates: list[dict], game_time: str) -> Optional[dict]:
        """From multiple candidate markets, pick the one closest to game_time.

        If game_time is empty or all candidates lack time info, returns the
        first candidate (same behavior as before).
        """
        if not candidates:
            return None
        if len(candidates) == 1 or not game_time:
            return candidates[0]
        return min(candidates, key=lambda m: self._time_distance(m, game_time))

    def find_all_markets_for_game(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        game_time: str = "",
    ) -> list[dict]:
        """Find all Polymarket markets for a given game.

        Args:
            game_time: Optional "HH:MM" (UTC) to disambiguate doubleheaders.
                       When provided, results are sorted by time proximity.
        """
        home_keyword = extract_school_keyword(home_team)
        away_keyword = extract_school_keyword(away_team)

        matches = []
        for market in self.markets:
            market_date = self._get_date(market)
            if market_date:
                date_diff = abs((market_date.date() - game_date.date()).days)
                if date_diff > 1:
                    continue

            if self._teams_match(market, home_keyword, away_keyword):
                matches.append(market)

        if game_time and len(matches) > 1:
            matches.sort(key=lambda m: self._time_distance(m, game_time))

        return matches

    def find_game_market(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        game_time: str = "",
    ) -> Optional[dict]:
        """Find the moneyline/game-winner market for a matchup."""
        candidates = self.find_all_markets_for_game(
            home_team, away_team, game_date, game_time=game_time,
        )
        game_markets = [m for m in candidates if self._get_market_type(m) == "game"]
        return self._pick_closest(game_markets, game_time)

    def find_spread_market(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        spread: float,
        game_time: str = "",
    ) -> Optional[dict]:
        """Find the best-matching spread market for a game."""
        candidates = self.find_all_markets_for_game(
            home_team, away_team, game_date, game_time=game_time,
        )
        spread_markets = [m for m in candidates if self._get_market_type(m) == "spread"]

        if not spread_markets:
            return None

        best = None
        best_diff = float("inf")
        for m in spread_markets:
            market_spread = self._get_spread_value(m)
            if market_spread is not None:
                diff = abs(market_spread - abs(spread))
                if diff < best_diff:
                    best_diff = diff
                    best = m
            elif best is None:
                best = m

        return best

    def get_market_prices(self, market: dict) -> dict:
        """Extract prices from a market dict (cached data, no API call).

        Returns dict with yes_price, no_price (0-100 scale), token_id, title.
        """
        yes_price, no_price = self._get_prices(market)
        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "token_id": self._get_token_id(market),
            "title": self._get_title(market),
        }

    def infer_yes_team(
        self,
        market: dict,
        home_team: str,
        away_team: str,
    ) -> Optional[str]:
        """Infer which team is YES for a game-winner market.

        Common patterns:
          "Will <Team> beat <Other>?" -> Team = YES
          "<TeamA> vs <TeamB>" -> first team = YES (convention)
        """
        title = self._get_title(market).lower()
        home_keyword = extract_school_keyword(home_team).lower()
        away_keyword = extract_school_keyword(away_team).lower()

        # "Will X beat Y?" or "X wins" -> X is YES
        beat_match = re.search(
            r"(?:will\s+)?(.+?)\s+(?:beat|defeat|win)", title, re.IGNORECASE
        )
        if beat_match:
            candidate = beat_match.group(1).strip().lower()
            if home_keyword in candidate:
                return home_team
            if away_keyword in candidate:
                return away_team

        # Outcomes list might tell us
        outcomes = market.get(self._f_outcomes, [])
        if isinstance(outcomes, list) and len(outcomes) >= 2:
            o0 = str(outcomes[0]).lower()
            o1 = str(outcomes[1]).lower()
            if home_keyword in o0:
                return home_team
            if away_keyword in o0:
                return away_team

        # Fallback: first team in title is YES
        team_a, team_b = self._extract_teams_from_title(market)
        if team_a:
            a_kw = extract_school_keyword(team_a).lower()
            if a_kw == home_keyword or home_keyword in a_kw:
                return home_team
            if a_kw == away_keyword or away_keyword in a_kw:
                return away_team

        return None
