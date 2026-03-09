"""Unit tests for Kalshi client transport/error helpers."""

import json

from kalshi.client import KalshiClient


class _DummyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        raise json.JSONDecodeError("invalid", "x", 0)


def test_get_handles_json_decode_error():
    client = KalshiClient(api_key="test")
    client.session.get = lambda *args, **kwargs: _DummyResponse()
    assert client._get("/markets", params={"status": "open"}) == {}


def test_get_auth_headers_skips_bad_signature(monkeypatch):
    client = KalshiClient(api_key="test")
    client.private_key = object()
    monkeypatch.setattr(client, "_sign_request", lambda method, path, timestamp: None)
    headers = client._get_auth_headers("GET", "/markets")
    assert "KALSHI-ACCESS-SIGNATURE" not in headers
    assert "KALSHI-ACCESS-TIMESTAMP" not in headers


def test_extract_market_prices_supports_dollars_fields():
    client = KalshiClient(api_key="test")
    prices = client._extract_market_prices(
        {
            "yes_ask_dollars": 0.43,
            "no_ask_dollars": 0.59,
            "last_price_dollars": 0.44,
        }
    )
    assert prices["yes_price"] == 43.0
    assert prices["no_price"] == 59.0
    assert prices["last_price"] == 44.0


def test_get_market_prices_falls_back_to_last_price(monkeypatch):
    client = KalshiClient(api_key="test")
    monkeypatch.setattr(
        client,
        "get_market",
        lambda ticker: {"ticker": ticker, "title": "Test", "last_price_dollars": 0.37},
    )
    prices = client.get_market_prices("KXTEST")
    assert prices["yes_price"] == 37.0
    assert prices["no_price"] == 63.0


def test_get_settlements_stops_at_total_limit(monkeypatch):
    client = KalshiClient(api_key="test")
    calls = []

    def _fake_get(endpoint, params=None):
        calls.append((endpoint, params))
        return {
            "settlements": [{"ticker": "A"}, {"ticker": "B"}, {"ticker": "C"}],
            "cursor": "next-page",
        }

    monkeypatch.setattr(client, "_get", _fake_get)
    settlements = client.get_settlements(limit=2)
    assert [s["ticker"] for s in settlements] == ["A", "B"]
    assert len(calls) == 1


def test_get_historical_markets_filters_series_prefix(monkeypatch):
    client = KalshiClient(api_key="test")

    monkeypatch.setattr(
        client,
        "_get",
        lambda endpoint, params=None: {
            "markets": [
                {"ticker": "KXNCAAMBGAME-EXAMPLE"},
                {"ticker": "KXOTHER-EXAMPLE"},
            ]
        },
    )

    markets = client.get_historical_markets(limit=10, series_ticker="KXNCAAMBGAME")
    assert [m["ticker"] for m in markets] == ["KXNCAAMBGAME-EXAMPLE"]


def test_get_market_any_falls_back_to_live_market(monkeypatch):
    client = KalshiClient(api_key="test")
    monkeypatch.setattr(client, "get_historical_market", lambda ticker: {})
    monkeypatch.setattr(client, "get_market", lambda ticker: {"ticker": ticker, "result": "yes"})
    market = client.get_market_any("KXNCAAMBGAME-EXAMPLE")
    assert market["result"] == "yes"
