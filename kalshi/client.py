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

        # Kalshi prices are in cents (0-100)
        # yes_ask is what you pay to buy YES -- use this for edge calculation
        yes_ask = market.get("yes_ask")
        yes_price = yes_ask if yes_ask is not None else market.get("last_price")
        no_ask = market.get("no_ask")
        no_price = no_ask if no_ask is not None else (100 - yes_price if yes_price is not None else None)

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "ticker": ticker,
            "title": market.get("title", ""),
        }

    def get_settlements(
        self,
        limit: int = 200,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[dict]:
        """Fetch settled positions from the user's portfolio.

        Args:
            limit: Max results per page (API max 200).
            ticker: Filter to a specific market ticker.
            min_ts: Minimum settlement timestamp (epoch seconds).
            max_ts: Maximum settlement timestamp (epoch seconds).

        Returns:
            List of settlement dicts from the API.
        """
        all_settlements: list[dict] = []
        cursor: Optional[str] = None

        while True:
            params: dict = {"limit": min(limit, 200)}
            if ticker:
                params["ticker"] = ticker
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts
            if cursor:
                params["cursor"] = cursor

            result = self._get("/portfolio/settlements", params)
            settlements = result.get("settlements", [])
            all_settlements.extend(settlements)

            cursor = result.get("cursor")
            if not cursor or len(settlements) < min(limit, 200):
                break

        return all_settlements
