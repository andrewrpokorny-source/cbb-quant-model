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

