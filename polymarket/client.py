"""Polymarket API client using the Gamma REST API with SOCKS5 proxy support.

All sports per-game markets (moneyline, spread, total) are served by the
Gamma API at https://gamma-api.polymarket.com. CLOB price data comes from
https://clob.polymarket.com.

Geo-restricted access is routed through a SOCKS5 proxy (SSH tunnel to a
non-blocked region). Requires ``requests[socks]`` (PySocks).
"""

import json
import os
from typing import Optional

import requests


# Sports tag IDs discovered from /sports endpoint.
# MLB tags='1,100639,100381' -> tag 100381 is MLB-specific.
# NCAAB tags='1,100149,100639' -> tag 100149 is NCAAB-specific.
SPORT_TAG_IDS = {
    "MLB": "100381",
    "NCAAB": "100149",
    "NBA": "10345",
    "NFL": "10187",
    "NHL": "10346",
    "WNBA": "10105",
}

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketClient:
    """Client for Polymarket Gamma + CLOB APIs with SOCKS5 proxy support."""

    def __init__(
        self,
        proxy_url: Optional[str] = None,
        timeout: int = 15,
    ):
        self.proxy_url = proxy_url or os.getenv("POLYMARKET_PROXY")
        self.timeout = timeout
        self.session = requests.Session()
        if self.proxy_url:
            self.session.proxies = {
                "https": self.proxy_url,
                "http": self.proxy_url,
            }

    def _get(self, url: str, params: Optional[dict] = None) -> dict | list:
        """GET request with proxy routing and error handling."""
        try:
            r = self.session.get(url, params=params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            print(f"      Polymarket API error: {e}")
            return []

    # -- Sports metadata --

    def get_sports(self) -> list[dict]:
        """List all supported sports with IDs and series info."""
        return self._get(f"{GAMMA_BASE}/sports")

    def get_sports_market_types(self) -> dict:
        """List valid sports market types."""
        return self._get(f"{GAMMA_BASE}/sports/market-types")

    # -- Events and markets --

    def get_events(
        self,
        tag_id: Optional[str] = None,
        active: bool = True,
        closed: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """Fetch events from Gamma API, optionally filtered by tag."""
        params = {"limit": str(limit)}
        if tag_id:
            params["tag_id"] = tag_id
        if active:
            params["active"] = "true"
        if not closed:
            params["closed"] = "false"
        result = self._get(f"{GAMMA_BASE}/events", params)
        return result if isinstance(result, list) else []

    def get_sports_events(self, league: str, limit: int = 50) -> list[dict]:
        """Fetch per-game sports events for a league.

        Uses the sport-specific tag ID to filter. Returns events with
        nested ``markets`` arrays containing moneyline, spread, and total
        markets.
        """
        tag_id = SPORT_TAG_IDS.get(league.upper())
        if not tag_id:
            print(f"      Unknown Polymarket league: {league}")
            return []
        return self.get_events(tag_id=tag_id, active=True, closed=False, limit=limit)

    def get_sports_game_markets(self, league: str, limit: int = 100) -> list[dict]:
        """Fetch per-game markets (flattened from events) for a league.

        Filters to events with ``teams`` and ``gameId`` fields (per-game
        events vs futures/props). Flattens the nested markets and attaches
        event-level metadata to each market dict.
        """
        events = self.get_sports_events(league, limit=limit)
        markets = []
        for event in events:
            # Per-game events have teams and a gameId
            if not event.get("teams") or not event.get("gameId"):
                continue
            event_meta = {
                "event_title": event.get("title", ""),
                "event_slug": event.get("slug", ""),
                "event_start_time": event.get("startTime"),
                "event_end_date": event.get("endDate"),
                "event_game_id": event.get("gameId"),
                "event_teams": event.get("teams", []),
            }
            for m in event.get("markets", []):
                market = {**m, **event_meta}
                # Parse JSON-encoded fields
                for field in ("clobTokenIds", "outcomePrices", "outcomes"):
                    val = market.get(field)
                    if isinstance(val, str):
                        try:
                            market[field] = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            pass
                markets.append(market)
        return markets

    # -- CLOB price data --

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token (0-1 scale)."""
        result = self._get(f"{CLOB_BASE}/midpoint", {"token_id": token_id})
        if isinstance(result, dict):
            mid = result.get("mid") or result.get("midpoint")
            if mid is not None:
                return float(mid)
        return None

    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get buy or sell price for a token (0-1 scale)."""
        result = self._get(f"{CLOB_BASE}/price", {"token_id": token_id, "side": side})
        if isinstance(result, dict):
            price = result.get("price")
            if price is not None:
                return float(price)
        return None

    def get_book(self, token_id: str) -> dict:
        """Get full order book for a token."""
        result = self._get(f"{CLOB_BASE}/book", {"token_id": token_id})
        return result if isinstance(result, dict) else {}

    # -- Market price helpers --

    def get_market_prices(self, token_id: str, title: str = "") -> dict:
        """Get YES/NO prices normalized to 0-100 cent scale.

        Polymarket native prices are 0-1 (probability). We multiply by 100
        to match the Kalshi convention used in ev_calculator.py.
        """
        mid = self.get_midpoint(token_id)
        if mid is not None:
            yes_price = round(mid * 100.0, 2)
            no_price = round(100.0 - yes_price, 2)
        else:
            buy = self.get_price(token_id, "BUY")
            if buy is not None:
                yes_price = round(buy * 100.0, 2)
                no_price = round(100.0 - yes_price, 2)
            else:
                yes_price = None
                no_price = None

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "token_id": token_id,
            "title": title,
        }

    # -- Position/portfolio data (public, by wallet address) --

    def get_positions(self, wallet_address: str) -> list[dict]:
        """Get open positions for a wallet address."""
        result = self._get(
            f"{GAMMA_BASE}/data/positions",
            {"address": wallet_address},
        )
        return result if isinstance(result, list) else []

    def get_closed_positions(self, wallet_address: str) -> list[dict]:
        """Get closed/settled positions for a wallet address."""
        result = self._get(
            f"{GAMMA_BASE}/data/closed-positions",
            {"address": wallet_address},
        )
        return result if isinstance(result, list) else []

    # -- Connectivity checks --

    def check_geoblock(self) -> dict:
        """Check if the current connection is geoblocked."""
        result = self._get("https://polymarket.com/api/geoblock")
        return result if isinstance(result, dict) else {}

    def is_ok(self) -> bool:
        """Check if the CLOB API is reachable."""
        try:
            r = self.session.get(f"{CLOB_BASE}/", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False
