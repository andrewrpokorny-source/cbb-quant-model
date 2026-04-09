"""Tests for doubleheader handling and timezone normalization in grade_predictions.py."""

import pytest

from grade_predictions import match_prediction_to_game


class TestDoubleheaderKeying:
    """When the same matchup appears twice, both games should be retrievable."""

    @pytest.fixture
    def doubleheader_games(self):
        """Simulate two games: (Yankees, Red Sox) on same day, different times."""
        return {
            ("New York Yankees", "Boston Red Sox"): {
                "home_score": 3,
                "away_score": 5,
                "spread": 0,
                "home_name": "New York Yankees",
                "away_name": "Boston Red Sox",
                "game_time": "13:10",  # Eastern afternoon
            },
            ("New York Yankees", "Boston Red Sox", 2): {
                "home_score": 7,
                "away_score": 2,
                "spread": 0,
                "home_name": "New York Yankees",
                "away_name": "Boston Red Sox",
                "game_time": "19:10",  # Eastern evening
            },
        }

    def test_single_game_matches_without_time(self):
        games = {
            ("Duke Blue Devils", "UNC Tar Heels"): {
                "home_score": 75,
                "away_score": 70,
                "spread": -3.5,
                "home_name": "Duke Blue Devils",
                "away_name": "UNC Tar Heels",
                "game_time": "19:00",
            },
        }
        result = match_prediction_to_game("UNC Tar Heels @ Duke Blue Devils", games)
        assert result is not None
        assert result["home_score"] == 75

    def test_doubleheader_without_time_returns_first(self, doubleheader_games):
        result = match_prediction_to_game(
            "Boston Red Sox @ New York Yankees",
            doubleheader_games,
        )
        assert result is not None
        assert result["home_score"] == 3  # First game

    def test_doubleheader_afternoon_game(self, doubleheader_games):
        result = match_prediction_to_game(
            "Boston Red Sox @ New York Yankees",
            doubleheader_games,
            pred_time="13:10",
        )
        assert result is not None
        assert result["home_score"] == 3  # Afternoon game

    def test_doubleheader_evening_game(self, doubleheader_games):
        result = match_prediction_to_game(
            "Boston Red Sox @ New York Yankees",
            doubleheader_games,
            pred_time="19:10",
        )
        assert result is not None
        assert result["home_score"] == 7  # Evening game

    def test_doubleheader_closest_time_match(self, doubleheader_games):
        # 18:00 is closer to 19:10 (70 min) than 13:10 (290 min)
        result = match_prediction_to_game(
            "Boston Red Sox @ New York Yankees",
            doubleheader_games,
            pred_time="18:00",
        )
        assert result is not None
        assert result["home_score"] == 7  # Evening game is closer

    def test_no_match_returns_none(self, doubleheader_games):
        result = match_prediction_to_game(
            "Chicago Cubs @ Los Angeles Dodgers",
            doubleheader_games,
        )
        assert result is None


class TestFuzzyMatching:
    """Fuzzy name matching still works with new keying."""

    def test_partial_name_match(self):
        games = {
            ("UConn Huskies", "Providence Friars"): {
                "home_score": 80,
                "away_score": 65,
                "spread": -7.5,
                "home_name": "UConn Huskies",
                "away_name": "Providence Friars",
                "game_time": "19:00",
            },
        }
        # Prediction uses short name, game uses full name
        result = match_prediction_to_game("Providence @ UConn Huskies", games)
        assert result is not None
        assert result["home_score"] == 80
