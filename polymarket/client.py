"""Polymarket API client wrapping the polymarket CLI binary."""

import json
import os
import subprocess
from typing import Optional


class PolymarketClient:
    """Client that wraps the Polymarket CLI for market data access.

    All HTTP requests are routed through a proxy (required for US users)
    by setting HTTPS_PROXY/HTTP_PROXY/ALL_PROXY in the subprocess env.
    """

    def __init__(
        self,
        cli_path: Optional[str] = None,
        proxy_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.cli_path = cli_path or os.getenv("POLYMARKET_CLI_PATH", "polymarket")
        self.proxy_url = proxy_url or os.getenv("POLYMARKET_PROXY")
        self.timeout = timeout

    def _run_cli(self, args: list[str]) -> dict | list:
        """Execute a polymarket CLI command and return parsed JSON.

        Prepends ``-o json`` so every command emits machine-readable output.
        Proxy env vars are injected into the subprocess environment.

        Returns ``{}`` (for object commands) or ``[]`` (for list commands)
        on any failure.
        """
        cmd = [self.cli_path, "-o", "json"] + args

        env = os.environ.copy()
        if self.proxy_url:
            env["HTTPS_PROXY"] = self.proxy_url
            env["HTTP_PROXY"] = self.proxy_url
            env["ALL_PROXY"] = self.proxy_url

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                if stderr:
                    print(f"      Polymarket CLI error: {stderr}")
                return [] if not result.stdout.strip() else {}

            stdout = result.stdout.strip()
            if not stdout:
                return []

            return json.loads(stdout)

        except FileNotFoundError:
            print("      Polymarket CLI not found. Install: brew tap Polymarket/polymarket-cli https://github.com/Polymarket/polymarket-cli && brew install polymarket")
            return []
        except subprocess.TimeoutExpired:
            print(f"      Polymarket CLI timed out after {self.timeout}s")
            return []
        except json.JSONDecodeError as e:
            print(f"      Polymarket CLI returned invalid JSON: {e}")
            return []

    # -- Read-only market browsing (no wallet needed) --

    def search_markets(self, query: str, limit: int = 50) -> list[dict]:
        """Search markets by text query."""
        return self._run_cli(["markets", "search", query, "--limit", str(limit)])

    def list_events(
        self,
        tag: Optional[str] = None,
        active: bool = True,
        limit: int = 50,
    ) -> list[dict]:
        """List events, optionally filtered by tag."""
        args = ["events", "list", "--limit", str(limit)]
        if tag:
            args += ["--tag", tag]
        if active:
            args += ["--active", "true"]
        return self._run_cli(args)

    def get_market(self, id_or_slug: str) -> dict:
        """Get a single market by numeric ID or slug."""
        result = self._run_cli(["markets", "get", str(id_or_slug)])
        return result if isinstance(result, dict) else {}

    def get_sports_markets(self, league: str, limit: int = 100) -> list[dict]:
        """List sports markets for a league (e.g. 'NCAAB', 'MLB', 'NBA')."""
        result = self._run_cli(
            ["sports", "list", "--league", league, "--limit", str(limit)]
        )
        return result if isinstance(result, list) else []

    def get_sports_market_types(self) -> list[dict]:
        """List available sports market types (moneyline, spread, total)."""
        result = self._run_cli(["sports", "market-types"])
        return result if isinstance(result, list) else []

    def get_sports_teams(self, league: str, limit: int = 100) -> list[dict]:
        """List teams for a league."""
        result = self._run_cli(
            ["sports", "teams", "--league", league, "--limit", str(limit)]
        )
        return result if isinstance(result, list) else []

    # -- CLOB price data (no wallet needed) --

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token (0-1 scale)."""
        result = self._run_cli(["clob", "midpoint", token_id])
        if isinstance(result, dict):
            mid = result.get("mid") or result.get("midpoint")
            if mid is not None:
                return float(mid)
        return None

    def get_price(self, token_id: str, side: str = "buy") -> Optional[float]:
        """Get buy or sell price for a token (0-1 scale)."""
        result = self._run_cli(["clob", "price", token_id, "--side", side])
        if isinstance(result, dict):
            price = result.get("price")
            if price is not None:
                return float(price)
        return None

    def get_spread(self, token_id: str) -> Optional[dict]:
        """Get bid-ask spread for a token."""
        result = self._run_cli(["clob", "spread", token_id])
        return result if isinstance(result, dict) else None

    def get_book(self, token_id: str) -> dict:
        """Get full order book for a token."""
        result = self._run_cli(["clob", "book", token_id])
        return result if isinstance(result, dict) else {}

    def get_fee_rate(self, token_id: str) -> Optional[float]:
        """Get fee rate for a token."""
        result = self._run_cli(["clob", "fee-rate", token_id])
        if isinstance(result, dict):
            rate = result.get("fee_rate") or result.get("feeRate")
            if rate is not None:
                return float(rate)
        return None

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
            buy = self.get_price(token_id, "buy")
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

    # -- Position/portfolio data (wallet address, public) --

    def get_positions(self, wallet_address: str) -> list[dict]:
        """Get open positions for a wallet address."""
        result = self._run_cli(["data", "positions", wallet_address])
        return result if isinstance(result, list) else []

    def get_closed_positions(self, wallet_address: str) -> list[dict]:
        """Get closed/settled positions for a wallet address."""
        result = self._run_cli(["data", "closed-positions", wallet_address])
        return result if isinstance(result, list) else []

    def get_trades(self, wallet_address: str, limit: int = 50) -> list[dict]:
        """Get recent trades for a wallet address."""
        result = self._run_cli(
            ["data", "trades", wallet_address, "--limit", str(limit)]
        )
        return result if isinstance(result, list) else []

    # -- Connectivity check --

    def check_geoblock(self) -> dict:
        """Check if the current connection is geoblocked."""
        result = self._run_cli(["clob", "geoblock"])
        return result if isinstance(result, dict) else {}

    def is_ok(self) -> bool:
        """Check if the CLOB API is reachable."""
        result = self._run_cli(["clob", "ok"])
        if isinstance(result, dict):
            return result.get("ok", False) or result.get("status") == "ok"
        return False
