"""Tests for polymarket.client CLI wrapper."""

import json
import subprocess

import pytest

from polymarket.client import PolymarketClient


class TestRunCli:
    """Tests for the core _run_cli subprocess wrapper."""

    def test_returns_parsed_json_list(self, monkeypatch):
        payload = [{"id": "1", "question": "Test?"}]
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 0, stdout=json.dumps(payload), stderr=""
            ),
        )
        client = PolymarketClient(cli_path="polymarket")
        result = client._run_cli(["markets", "list"])
        assert result == payload

    def test_returns_parsed_json_dict(self, monkeypatch):
        payload = {"mid": "0.55"}
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 0, stdout=json.dumps(payload), stderr=""
            ),
        )
        client = PolymarketClient()
        result = client._run_cli(["clob", "midpoint", "TOKEN"])
        assert result == payload

    def test_returns_empty_list_on_cli_not_found(self, monkeypatch):
        def _raise(*a, **kw):
            raise FileNotFoundError("polymarket not found")
        monkeypatch.setattr(subprocess, "run", _raise)
        client = PolymarketClient()
        assert client._run_cli(["markets", "list"]) == []

    def test_returns_empty_list_on_timeout(self, monkeypatch):
        def _raise(*a, **kw):
            raise subprocess.TimeoutExpired("polymarket", 30)
        monkeypatch.setattr(subprocess, "run", _raise)
        client = PolymarketClient(timeout=1)
        assert client._run_cli(["clob", "ok"]) == []

    def test_returns_empty_on_invalid_json(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 0, stdout="not json {{{", stderr=""
            ),
        )
        client = PolymarketClient()
        assert client._run_cli(["markets", "list"]) == []

    def test_returns_empty_list_on_nonzero_exit_no_stdout(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(
                a[0], 1, stdout="", stderr="error"
            ),
        )
        client = PolymarketClient()
        assert client._run_cli(["clob", "ok"]) == []


class TestProxyEnv:
    """Verify proxy env vars are set in subprocess calls."""

    def test_proxy_env_vars_set(self, monkeypatch):
        captured_env = {}

        def fake_run(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return subprocess.CompletedProcess(args[0], 0, stdout="[]", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        client = PolymarketClient(proxy_url="socks5://127.0.0.1:8080")
        client._run_cli(["markets", "list"])

        assert captured_env.get("HTTPS_PROXY") == "socks5://127.0.0.1:8080"
        assert captured_env.get("HTTP_PROXY") == "socks5://127.0.0.1:8080"
        assert captured_env.get("ALL_PROXY") == "socks5://127.0.0.1:8080"

    def test_no_proxy_when_none(self, monkeypatch):
        captured_env = {}

        def fake_run(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return subprocess.CompletedProcess(args[0], 0, stdout="[]", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.delenv("POLYMARKET_PROXY", raising=False)
        client = PolymarketClient(proxy_url=None)
        client._run_cli(["markets", "list"])

        # Should not have injected proxy vars (beyond what's in os.environ)
        assert "ALL_PROXY" not in captured_env or captured_env["ALL_PROXY"] != "socks5://127.0.0.1:8080"


class TestMarketPrices:
    """Tests for price normalization to 0-100 cent scale."""

    def test_midpoint_converted_to_cents(self, monkeypatch):
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            # First call: midpoint
            return subprocess.CompletedProcess(
                args[0], 0,
                stdout=json.dumps({"mid": "0.43"}),
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        client = PolymarketClient()
        prices = client.get_market_prices("TOKEN123", title="Test")

        assert prices["yes_price"] == 43.0
        assert prices["no_price"] == 57.0
        assert prices["token_id"] == "TOKEN123"
        assert prices["title"] == "Test"

    def test_fallback_to_buy_price(self, monkeypatch):
        calls = []

        def fake_run(*args, **kwargs):
            cmd = args[0]
            calls.append(cmd)
            if "midpoint" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")
            if "price" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=json.dumps({"price": "0.70"}), stderr=""
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        client = PolymarketClient()
        prices = client.get_market_prices("TOKEN")
        assert prices["yes_price"] == 70.0
        assert prices["no_price"] == 30.0

    def test_returns_none_when_no_prices(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout="{}", stderr=""),
        )
        client = PolymarketClient()
        prices = client.get_market_prices("TOKEN")
        assert prices["yes_price"] is None
        assert prices["no_price"] is None
