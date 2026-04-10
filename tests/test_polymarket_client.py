"""Tests for polymarket.client Gamma + CLOB API wrapper."""

import json

import pytest
import requests

from polymarket.client import PolymarketClient, GAMMA_BASE, CLOB_BASE, SPORT_TAG_IDS


class _FakeResponse:
    """Minimal requests.Response stand-in."""

    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class TestGet:
    """Core _get method error handling."""

    def test_returns_parsed_json(self, monkeypatch):
        payload = [{"id": "1", "title": "Test"}]
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, **kw: _FakeResponse(payload),
        )
        client = PolymarketClient(proxy_url=None)
        assert client._get(f"{GAMMA_BASE}/events") == payload

    def test_returns_empty_on_request_error(self, monkeypatch):
        def _raise(*a, **kw):
            raise requests.ConnectionError("no connection")
        monkeypatch.setattr(requests.Session, "get", _raise)
        client = PolymarketClient(proxy_url=None)
        assert client._get(f"{GAMMA_BASE}/events") == []

    def test_returns_empty_on_http_error(self, monkeypatch):
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, **kw: _FakeResponse({}, status_code=500),
        )
        client = PolymarketClient(proxy_url=None)
        assert client._get(f"{GAMMA_BASE}/events") == []


class TestProxyConfig:
    """Verify proxy configuration on session."""

    def test_proxy_set_from_constructor(self):
        client = PolymarketClient(proxy_url="socks5h://127.0.0.1:8080")
        assert client.session.proxies["https"] == "socks5h://127.0.0.1:8080"
        assert client.session.proxies["http"] == "socks5h://127.0.0.1:8080"

    def test_no_proxy_when_none(self, monkeypatch):
        monkeypatch.delenv("POLYMARKET_PROXY", raising=False)
        client = PolymarketClient(proxy_url=None)
        assert not client.session.proxies


class TestGetSportsGameMarkets:
    """Test game market fetching and flattening."""

    def test_flattens_event_markets(self, monkeypatch):
        events = [{
            "title": "Yankees vs Red Sox",
            "gameId": "12345",
            "startTime": "2026-04-10T23:00:00Z",
            "slug": "mlb-nyy-bos-2026-04-10",
            "teams": [
                {"name": "New York Yankees", "abbreviation": "nyy"},
                {"name": "Boston Red Sox", "abbreviation": "bos"},
            ],
            "markets": [
                {
                    "question": "New York Yankees vs. Boston Red Sox",
                    "outcomes": '["New York Yankees", "Boston Red Sox"]',
                    "outcomePrices": '["0.55", "0.45"]',
                    "clobTokenIds": '["token_yes", "token_no"]',
                    "conditionId": "0xabc",
                    "slug": "mlb-nyy-bos-2026-04-10",
                },
                {
                    "question": "Spread: Boston Red Sox (-1.5)",
                    "outcomes": '["Boston Red Sox", "New York Yankees"]',
                    "outcomePrices": '["0.48", "0.52"]',
                    "clobTokenIds": '["tok_sp_yes", "tok_sp_no"]',
                    "conditionId": "0xdef",
                    "slug": "mlb-nyy-bos-2026-04-10-spread-home-1pt5",
                },
            ],
        }]
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, **kw: _FakeResponse(events),
        )
        client = PolymarketClient(proxy_url=None)
        markets = client.get_sports_game_markets("MLB")

        assert len(markets) == 2
        # JSON strings should be parsed
        assert markets[0]["outcomes"] == ["New York Yankees", "Boston Red Sox"]
        assert markets[0]["outcomePrices"] == ["0.55", "0.45"]
        assert markets[0]["clobTokenIds"] == ["token_yes", "token_no"]
        # Event metadata should be attached
        assert markets[0]["event_teams"][0]["name"] == "New York Yankees"
        assert markets[0]["event_start_time"] == "2026-04-10T23:00:00Z"

    def test_skips_non_game_events(self, monkeypatch):
        events = [
            {"title": "MLB World Series Champion 2026", "markets": [{"question": "Will Yankees win?"}]},
            {"title": "Some futures", "gameId": None, "teams": None, "markets": []},
        ]
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, **kw: _FakeResponse(events),
        )
        client = PolymarketClient(proxy_url=None)
        markets = client.get_sports_game_markets("MLB")
        assert len(markets) == 0

    def test_unknown_league_returns_empty(self, monkeypatch):
        client = PolymarketClient(proxy_url=None)
        assert client.get_sports_game_markets("CURLING") == []


class TestMarketPrices:
    """Price normalization to 0-100 cent scale."""

    def test_midpoint_converted_to_cents(self, monkeypatch):
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, **kw: _FakeResponse({"mid": "0.43"}),
        )
        client = PolymarketClient(proxy_url=None)
        prices = client.get_market_prices("TOKEN123", title="Test")
        assert prices["yes_price"] == 43.0
        assert prices["no_price"] == 57.0

    def test_fallback_to_buy_price(self, monkeypatch):
        call_count = [0]
        def fake_get(self, url, **kw):
            call_count[0] += 1
            if "midpoint" in url:
                return _FakeResponse({})
            if "price" in url:
                return _FakeResponse({"price": "0.70"})
            return _FakeResponse({})
        monkeypatch.setattr(requests.Session, "get", fake_get)
        client = PolymarketClient(proxy_url=None)
        prices = client.get_market_prices("TOKEN")
        assert prices["yes_price"] == 70.0
        assert prices["no_price"] == 30.0

    def test_returns_none_when_no_prices(self, monkeypatch):
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, **kw: _FakeResponse({}),
        )
        client = PolymarketClient(proxy_url=None)
        prices = client.get_market_prices("TOKEN")
        assert prices["yes_price"] is None
        assert prices["no_price"] is None


class TestSportTagIds:
    """Verify sport tag ID mapping."""

    def test_mlb_tag_id(self):
        assert SPORT_TAG_IDS["MLB"] == "100381"

    def test_ncaab_tag_id(self):
        assert SPORT_TAG_IDS["NCAAB"] == "100149"


class TestConnectivity:
    """Geoblock and health checks."""

    def test_geoblock_returns_dict(self, monkeypatch):
        monkeypatch.setattr(
            requests.Session, "get",
            lambda self, url, **kw: _FakeResponse({"blocked": False, "country": "ID"}),
        )
        client = PolymarketClient(proxy_url=None)
        result = client.check_geoblock()
        assert result["blocked"] is False

    def test_is_ok_returns_true(self, monkeypatch):
        resp = _FakeResponse("OK")
        resp.status_code = 200
        monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: resp)
        client = PolymarketClient(proxy_url=None)
        assert client.is_ok() is True
