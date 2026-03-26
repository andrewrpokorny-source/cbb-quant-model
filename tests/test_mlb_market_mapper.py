"""Tests for MLB Kalshi market ticker matching."""

from datetime import datetime

import pytest

from kalshi.mlb_market_mapper import MLBMarketMapper, MLB_ABBREVIATIONS, ABBR_TO_TEAM


def _make_market(ticker, title=""):
    return {"ticker": ticker, "title": title, "yes_ask": 55}


class TestParseTickerTeams:

    def test_standard_game_ticker(self):
        mapper = MLBMarketMapper([])
        away, home = mapper._parse_ticker_teams("KXMLBGAME-26MAR282040DETSD-SD")
        assert away == "DET"
        assert home == "SD"

    def test_three_letter_abbreviations(self):
        mapper = MLBMarketMapper([])
        away, home = mapper._parse_ticker_teams("KXMLBGAME-26MAR281910LAAHOU-LAA")
        assert away == "LAA"
        assert home == "HOU"

    def test_mixed_length_abbreviations(self):
        mapper = MLBMarketMapper([])
        away, home = mapper._parse_ticker_teams("KXMLBGAME-26MAR271915KCATL-ATL")
        assert away == "KC"
        assert home == "ATL"

    def test_spread_ticker(self):
        mapper = MLBMarketMapper([])
        away, home = mapper._parse_ticker_teams("KXMLBSPREAD-26MAR271915KCATL-KC4")
        assert away == "KC"
        assert home == "ATL"

    def test_invalid_ticker_returns_empty(self):
        mapper = MLBMarketMapper([])
        away, home = mapper._parse_ticker_teams("GARBAGE")
        assert away == ""
        assert home == ""

    def test_short_ticker_returns_empty(self):
        mapper = MLBMarketMapper([])
        away, home = mapper._parse_ticker_teams("KXMLB-26-WSH")
        assert away == ""
        assert home == ""


class TestFindMarket:

    def test_finds_exact_match(self):
        markets = [_make_market("KXMLBGAME-26MAR282040DETSD-SD")]
        mapper = MLBMarketMapper(markets)
        result = mapper.find_market(
            "San Diego Padres", "Detroit Tigers",
            datetime(2026, 3, 28), "GAME"
        )
        assert result is not None
        assert result["ticker"] == "KXMLBGAME-26MAR282040DETSD-SD"

    def test_returns_none_for_wrong_date(self):
        markets = [_make_market("KXMLBGAME-26MAR282040DETSD-SD")]
        mapper = MLBMarketMapper(markets)
        result = mapper.find_market(
            "San Diego Padres", "Detroit Tigers",
            datetime(2026, 3, 29), "GAME"
        )
        assert result is None

    def test_no_false_match_on_substring_abbreviations(self):
        """KC should NOT match inside OAK+CIN concatenation."""
        markets = [_make_market("KXMLBGAME-26MAR282040OAKCIN-CIN")]
        mapper = MLBMarketMapper(markets)
        result = mapper.find_market(
            "Atlanta Braves", "Kansas City Royals",
            datetime(2026, 3, 28), "GAME"
        )
        assert result is None

    def test_sf_no_false_match(self):
        """SF should NOT match inside a ticker for different teams."""
        markets = [_make_market("KXMLBGAME-26MAR282040SFCOL-SF")]
        mapper = MLBMarketMapper(markets)
        # Searching for a completely different game
        result = mapper.find_market(
            "Houston Astros", "Los Angeles Angels",
            datetime(2026, 3, 28), "GAME"
        )
        assert result is None

    def test_falls_back_to_title_match(self):
        """When ticker parsing fails, fall back to title matching."""
        markets = [_make_market(
            "KXMLBGAME-WEIRD-FORMAT",
            title="Detroit Tigers vs San Diego Padres Winner?"
        )]
        mapper = MLBMarketMapper(markets)
        result = mapper.find_market(
            "San Diego Padres", "Detroit Tigers",
            datetime(2026, 3, 28), "GAME"
        )
        assert result is not None

    def test_unknown_team_returns_none(self):
        markets = [_make_market("KXMLBGAME-26MAR282040DETSD-SD")]
        mapper = MLBMarketMapper(markets)
        result = mapper.find_market(
            "Fake Team", "Detroit Tigers",
            datetime(2026, 3, 28), "GAME"
        )
        assert result is None

    def test_filters_by_market_type(self):
        markets = [
            _make_market("KXMLBGAME-26MAR282040DETSD-SD"),
            _make_market("KXMLBSPREAD-26MAR282040DETSD-SD4"),
        ]
        mapper = MLBMarketMapper(markets)
        game = mapper.find_market(
            "San Diego Padres", "Detroit Tigers",
            datetime(2026, 3, 28), "GAME"
        )
        spread = mapper.find_market(
            "San Diego Padres", "Detroit Tigers",
            datetime(2026, 3, 28), "SPREAD"
        )
        assert "GAME" in game["ticker"]
        assert "SPREAD" in spread["ticker"]


class TestGetYesTeam:

    def test_resolves_abbreviation_to_full_name(self):
        mapper = MLBMarketMapper([])
        assert mapper.get_yes_team("KXMLBGAME-26MAR282040DETSD-SD") == "San Diego Padres"
        assert mapper.get_yes_team("KXMLBGAME-26MAR282040DETSD-DET") == "Detroit Tigers"
        assert mapper.get_yes_team("KXMLBGAME-26MAR281915NYYSF-NYY") == "New York Yankees"

    def test_returns_none_for_unknown_suffix(self):
        mapper = MLBMarketMapper([])
        assert mapper.get_yes_team("KXMLBGAME-26MAR282040DETSD-XYZ") is None

    def test_returns_none_for_no_dash(self):
        mapper = MLBMarketMapper([])
        assert mapper.get_yes_team("NOSEPARATOR") is None


class TestAbbreviationCompleteness:

    def test_all_30_teams_mapped(self):
        assert len(MLB_ABBREVIATIONS) == 30

    def test_reverse_lookup_consistent(self):
        for name, abbr in MLB_ABBREVIATIONS.items():
            assert ABBR_TO_TEAM[abbr] == name
