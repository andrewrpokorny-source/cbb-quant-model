"""Tests for Kalshi GAME market helper behavior in predict.py."""

from datetime import datetime

import pytest

from betting import calculate_edge
from predict import (
    _infer_yes_team_from_game_market,
    get_kalshi_game_edge,
    get_kalshi_game_live_rating,
)


class _StubMapper:
    def __init__(self, markets):
        self._markets = markets

    def find_all_markets_for_game(self, home_team, away_team, game_date):
        return self._markets

    def get_market_prices(self, market):
        return {
            "yes_price": market.get("yes_ask"),
            "no_price": market.get("no_ask"),
            "ticker": market.get("ticker"),
            "title": market.get("title", ""),
        }


class _StubClient:
    def __init__(self, prices_by_ticker):
        self._prices_by_ticker = prices_by_ticker

    def get_market_prices(self, ticker):
        return self._prices_by_ticker[ticker]


def test_infer_yes_team_from_rules():
    market = {
        "title": "Who wins?",
        "rules_primary": "This market resolves to YES if UConn wins the game.",
    }
    yes_team = _infer_yes_team_from_game_market(
        market,
        home_team="UConn Huskies",
        away_team="Providence Friars",
    )
    assert yes_team == "UConn Huskies"


def test_infer_yes_team_from_title_fallback():
    market = {
        "title": "Will Providence win tonight?",
        "rules_primary": "",
    }
    yes_team = _infer_yes_team_from_game_market(
        market,
        home_team="UConn Huskies",
        away_team="Providence Friars",
    )
    assert yes_team == "Providence Friars"


def test_infer_yes_team_disambiguates_substring_teams():
    """Virginia Tech should not be confused with Virginia."""
    market = {
        "title": "Virginia Tech vs Virginia",
        "rules_primary": "This market resolves to YES if Virginia Tech wins the game.",
    }
    yes_team = _infer_yes_team_from_game_market(
        market,
        home_team="Virginia Cavaliers",
        away_team="Virginia Tech Hokies",
    )
    assert yes_team == "Virginia Tech Hokies"


def test_infer_yes_team_exact_match_not_confused_by_longer():
    """Virginia should match correctly even when Virginia Tech is the other team."""
    market = {
        "title": "Virginia Tech vs Virginia",
        "rules_primary": "This market resolves to YES if Virginia wins the game.",
    }
    yes_team = _infer_yes_team_from_game_market(
        market,
        home_team="Virginia Cavaliers",
        away_team="Virginia Tech Hokies",
    )
    assert yes_team == "Virginia Cavaliers"


def test_infer_yes_team_abbreviated_name():
    """Kansas St. should resolve to Kansas State, not Kansas."""
    market = {
        "title": "Kansas St. at Kansas Winner?",
        "rules_primary": "If Kansas St. wins the Kansas St. at Kansas men's college basketball game.",
    }
    yes_team = _infer_yes_team_from_game_market(
        market,
        home_team="Kansas Jayhawks",
        away_team="Kansas State Wildcats",
    )
    assert yes_team == "Kansas State Wildcats"


def test_infer_yes_team_full_name_not_confused_by_abbreviation():
    """Kansas should resolve correctly when Kansas State is the opponent."""
    market = {
        "title": "Kansas St. at Kansas Winner?",
        "rules_primary": "If Kansas wins the Kansas St. at Kansas men's college basketball game.",
    }
    yes_team = _infer_yes_team_from_game_market(
        market,
        home_team="Kansas Jayhawks",
        away_team="Kansas State Wildcats",
    )
    assert yes_team == "Kansas Jayhawks"


def test_get_kalshi_game_edge_selects_best_side():
    market = {
        "ticker": "KXNCAAWBGAME-EXAMPLE",
        "title": "Resolves to YES if UConn wins",
        "rules_primary": "Resolves to YES if UConn wins.",
    }
    mapper = _StubMapper([market])
    client = _StubClient(
        {
            "KXNCAAWBGAME-EXAMPLE": {
                "yes_price": 60,
                "no_price": 30,
                "ticker": "KXNCAAWBGAME-EXAMPLE",
                "title": "Resolves to YES if UConn wins",
            }
        }
    )

    result = get_kalshi_game_edge(
        client=client,
        mapper=mapper,
        home_team="UConn Huskies",
        away_team="Providence Friars",
        game_date=datetime(2026, 2, 27),
        model_home_win_prob=0.4,
    )

    # YES maps to home team; with P(home)=0.4, NO has probability 0.6 and is better at 30c.
    # Fee-adjusted implied prob at 30c = 0.3 + 0.07*0.3*0.7 = 0.3147
    assert result["Kalshi_Side"] == "NO"
    assert result["Picked_Team"] == "Providence Friars"
    assert result["Edge"] == pytest.approx(calculate_edge(0.6, 0.3147))
    assert result["Kalshi_Yes_Team"] == "UConn Huskies"
    assert result["Rating"] == "STRONG"


def test_get_kalshi_game_live_rating_rejects_extreme_tail_longshot():
    rating = get_kalshi_game_live_rating(
        edge=calculate_edge(0.2862247836, 0.010693),
        side_prob=0.2862247836,
        price_cents=1,
    )
    assert rating == "PASS"


def test_get_kalshi_game_live_rating_requires_probability_and_price_bands():
    assert get_kalshi_game_live_rating(edge=0.09, side_prob=0.56, price_cents=55) == "STRONG"
    assert get_kalshi_game_live_rating(edge=0.05, side_prob=0.53, price_cents=60) == "GOOD"
    assert get_kalshi_game_live_rating(edge=0.09, side_prob=0.54, price_cents=55) == "GOOD"
    assert get_kalshi_game_live_rating(edge=0.05, side_prob=0.53, price_cents=10) == "PASS"
