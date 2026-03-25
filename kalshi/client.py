"""Kalshi API client for fetching prediction market data."""

import os
import json
import base64
import time
import requests
from typing import Optional
from datetime import datetime


class KalshiClient:
    """Client for Kalshi prediction market API with RSA signature auth."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
    ):
        """
        Initialize client with API key and optional private key for signing.

        Args:
            api_key: Kalshi API key (or set KALSHI_API_KEY env var)
            private_key_path: Path to RSA private key file (or set KALSHI_PRIVATE_KEY_PATH env var)
        """
        self.api_key = api_key or os.getenv("KALSHI_API_KEY")
        self.private_key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        self.private_key = None
        self.session = requests.Session()

        # Load private key if path provided
        if self.private_key_path and os.path.exists(self.private_key_path):
            self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key from file."""
        try:
            from cryptography.hazmat.primitives import serialization
            with open(self.private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                )
            print(f"      Loaded Kalshi private key")
        except ImportError:
            print("      cryptography package not installed, RSA signing disabled")
        except Exception as e:
            print(f"      Failed to load private key: {e}")

    def _sign_request(self, method: str, path: str, timestamp: str) -> Optional[str]:
        """
        Sign request with RSA private key.

        Returns base64-encoded signature.
        """
        if not self.private_key:
            return None

        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            # Message format: timestamp + method + path
            message = f"{timestamp}{method}{path}".encode()

            signature = self.private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            print(f"      Signing error: {e}")
            return None

    def _get_auth_headers(self, method: str, path: str) -> dict:
        """Get authentication headers for request."""
        headers = {}

        if self.api_key:
            headers["KALSHI-ACCESS-KEY"] = self.api_key

        if self.private_key:
            timestamp = str(int(time.time() * 1000))
            # Signing path must be the full URL path including the API prefix
            sign_path = f"/trade-api/v2{path}"
            signature = self._sign_request(method, sign_path, timestamp)
            if signature:
                headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
                headers["KALSHI-ACCESS-SIGNATURE"] = signature
            else:
                print("      Skipping signed headers due to signature generation failure")

        return headers

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request to Kalshi API."""
        url = f"{self.BASE_URL}{endpoint}"
        headers = self._get_auth_headers("GET", endpoint)

        try:
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            print(f"      Kalshi API error: {e}")
            return {}

    @staticmethod
    def _to_float(value):
        """Convert a value to float, returning None on failure."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_market_prices(self, market: dict) -> dict:
        """Extract fee-model prices from a market payload across old/new formats."""
        if not market:
            return {"yes_price": None, "no_price": None, "last_price": None}

        # Prefer *_dollars fields (0-1 scale) and convert to cents;
        # fall back to legacy cent-denominated fields used as-is.
        yes_dollars = self._to_float(market.get("yes_ask_dollars"))
        no_dollars = self._to_float(market.get("no_ask_dollars"))
        last_dollars = self._to_float(market.get("last_price_dollars"))

        if yes_dollars is not None:
            yes_price = round(yes_dollars * 100.0, 4)
        else:
            yes_price = self._to_float(market.get("yes_ask"))

        if no_dollars is not None:
            no_price = round(no_dollars * 100.0, 4)
        else:
            no_price = self._to_float(market.get("no_ask"))

        if last_dollars is not None:
            last_price = round(last_dollars * 100.0, 4)
        else:
            last_price = self._to_float(market.get("last_price"))


        if yes_price is None:
            yes_price = last_price
        if no_price is None and yes_price is not None:
            no_price = round(100.0 - yes_price, 4)

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "last_price": last_price,
        }

    def search_markets(
        self,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: str = "open",
        limit: int = 100,
    ) -> list:
        """
        Search for markets.

        Args:
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker (e.g., 'NCAAB' for college basketball)
            status: Market status ('open', 'closed', 'settled')
            limit: Max results to return

        Returns:
            List of market dictionaries
        """
        params = {"status": status, "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker

        result = self._get("/markets", params)
        return result.get("markets", [])

    def get_market(self, ticker: str) -> dict:
        """
        Get details for a specific market.

        Args:
            ticker: Market ticker (e.g., 'NCAAB-DUK-UNC-2026-01-20')

        Returns:
            Market details dictionary
        """
        result = self._get(f"/markets/{ticker}")
        return result.get("market", {})

    def get_orderbook(self, ticker: str) -> dict:
        """
        Get orderbook (current bid/ask prices) for a market.

        Args:
            ticker: Market ticker

        Returns:
            Orderbook with 'yes' and 'no' price info
        """
        result = self._get(f"/markets/{ticker}/orderbook")
        return result.get("orderbook", {})

    def get_ncaab_markets(self) -> list:
        """
        Get all open NCAAB (college basketball) spread markets.

        Kalshi uses ticker format: KXNCAAMBSPREAD-26JAN21LAFBU

        Returns:
            List of NCAAB market dictionaries
        """
        return self.get_college_basketball_markets(league="mens")

    def get_college_basketball_markets(self, league: str = "mens") -> list:
        """
        Get all open college basketball markets for a target league.

        Args:
            league: 'mens' or 'womens' (aliases: men/women/m/w)

        Returns:
            List of Kalshi market dictionaries.
        """
        key = str(league or "mens").strip().lower()
        if key in {"women", "womens", "w"}:
            canonical = "womens"
        else:
            canonical = "mens"

        # Known Kalshi series ticker families.
        # Men's: KXNCAAMB*
        # Women's: KXNCAAWB* (observed game markets) with potential spread/total variants.
        series_by_league = {
            "mens": [
                "KXNCAAMBSPREAD",
                "KXNCAAMBGAME",
                "KXNCAAMBTOTAL",
            ],
            "womens": [
                "KXNCAAWBSPREAD",
                "KXNCAAWBGAME",
                "KXNCAAWBTOTAL",
            ],
        }
        prefix_by_league = {
            "mens": "KXNCAAMB",
            "womens": "KXNCAAWB",
        }

        markets = []
        for series in series_by_league[canonical]:
            result = self.search_markets(series_ticker=series, status="open", limit=200)
            if result:
                markets.extend(result)

        if not markets:
            all_markets = self.search_markets(status="open", limit=1000)
            prefix = prefix_by_league[canonical]
            markets = [m for m in all_markets if m.get("ticker", "").startswith(prefix)]

        return markets

    def get_mlb_markets(self) -> list:
        """Get all open MLB markets from Kalshi.

        Returns:
            List of Kalshi market dictionaries for MLB game/spread/total markets.
        """
        series_tickers = [
            "KXMLBGAME",
            "KXMLBSPREAD",
            "KXMLBTOTAL",
        ]

        markets = []
        for series in series_tickers:
            result = self.search_markets(series_ticker=series, status="open", limit=200)
            if result:
                markets.extend(result)

        if not markets:
            all_markets = self.search_markets(status="open", limit=1000)
            markets = [m for m in all_markets if m.get("ticker", "").startswith("KXMLB")]

        return markets

    def get_market_prices(self, ticker: str) -> dict:
        """
        Get current Yes/No prices for a market.

        Args:
            ticker: Market ticker

        Returns:
            Dict with 'yes_price' and 'no_price' (0-100 scale)
        """
        market = self.get_market(ticker)
        if not market:
            return {"yes_price": None, "no_price": None}
        prices = self._extract_market_prices(market)

        return {
            "yes_price": prices["yes_price"],
            "no_price": prices["no_price"],
            "ticker": ticker,
            "title": market.get("title", ""),
        }

    def get_historical_markets(
        self,
        limit: int = 200,
        ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
    ) -> list[dict]:
        """Fetch historical markets from Kalshi's historical API."""
        all_markets: list[dict] = []
        cursor: Optional[str] = None
        page_size = min(limit, 200)

        while True:
            params: dict = {"limit": page_size}
            if ticker:
                params["ticker"] = ticker
            if event_ticker:
                params["event_ticker"] = event_ticker
            if series_ticker:
                params["series_ticker"] = series_ticker
            if min_close_ts is not None:
                params["min_close_ts"] = min_close_ts
            if max_close_ts is not None:
                params["max_close_ts"] = max_close_ts
            if cursor:
                params["cursor"] = cursor

            result = self._get("/historical/markets", params)
            if not result:
                break

            raw_markets = result.get("markets", [])
            markets = raw_markets
            if series_ticker:
                markets = [m for m in raw_markets if m.get("ticker", "").startswith(series_ticker)]
            all_markets.extend(markets)
            if len(all_markets) >= limit:
                return all_markets[:limit]
            cursor = result.get("cursor")
            if not cursor or len(raw_markets) < page_size:
                break

        return all_markets

    def get_historical_market(self, ticker: str) -> dict:
        """Fetch one historical market record."""
        result = self._get(f"/historical/markets/{ticker}")
        return result.get("market", result)

    def get_market_any(self, ticker: str) -> dict:
        """Fetch a market from historical storage first, then the live endpoint."""
        market = self.get_historical_market(ticker)
        if market:
            return market
        return self.get_market(ticker)

    def get_historical_market_candlesticks(
        self,
        ticker: str,
        *,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        period_interval: Optional[int] = None,
    ) -> list[dict]:
        """Fetch historical candlesticks for a market."""
        params: dict = {}
        if start_ts is not None:
            params["start_ts"] = start_ts
        if end_ts is not None:
            params["end_ts"] = end_ts
        if period_interval is not None:
            params["period_interval"] = period_interval
        result = self._get(f"/historical/markets/{ticker}/candlesticks", params or None)
        return result.get("candlesticks", [])

    def get_historical_trades(
        self,
        *,
        ticker: Optional[str] = None,
        limit: int = 1000,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[dict]:
        """Fetch historical trades from Kalshi's historical API."""
        all_trades: list[dict] = []
        cursor: Optional[str] = None
        page_size = min(limit, 1000)

        while True:
            params: dict = {"limit": page_size}
            if ticker:
                params["ticker"] = ticker
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts
            if cursor:
                params["cursor"] = cursor

            result = self._get("/historical/trades", params)
            if not result:
                break

            trades = result.get("trades", [])
            all_trades.extend(trades)
            if len(all_trades) >= limit:
                return all_trades[:limit]
            cursor = result.get("cursor")
            if not cursor or len(trades) < page_size:
                break

        return all_trades

    def get_settlements(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[dict]:
        """Fetch settled positions from the user's portfolio.

        Paginates through all results (API max 200 per page).

        Args:
            ticker: Filter to a specific market ticker.
            min_ts: Minimum settlement timestamp (epoch seconds).
            max_ts: Maximum settlement timestamp (epoch seconds).

        Returns:
            List of settlement dicts from the API.
        """
        all_settlements: list[dict] = []
        cursor: Optional[str] = None

        while True:
            params: dict = {"limit": 200}
            if ticker:
                params["ticker"] = ticker
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts
            if cursor:
                params["cursor"] = cursor

            result = self._get("/portfolio/settlements", params)
            if not result:
                raise RuntimeError("Kalshi API request failed fetching settlements")
            settlements = result.get("settlements", [])
            all_settlements.extend(settlements)

            cursor = result.get("cursor")
            if not cursor or len(settlements) < 200:
                break

        return all_settlements
