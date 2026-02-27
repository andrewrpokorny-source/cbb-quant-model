"""Unit tests for GAME pick parsing/grading helpers in grade_predictions.py."""

from grade_predictions import parse_game_pick, grade_game_pick


def test_parse_game_pick_ml_yes():
    assert parse_game_pick("UConn ML YES") == {"team": "UConn", "side": "YES"}


def test_parse_game_pick_ml_default_yes():
    assert parse_game_pick("UConn ML") == {"team": "UConn", "side": "YES"}


def test_grade_game_pick_yes_loss():
    game_pick = {"team": "UConn", "side": "YES"}
    game_result = {
        "home_name": "UConn Huskies",
        "away_name": "Providence Friars",
        "home_score": 60,
        "away_score": 70,
    }
    assert grade_game_pick(game_pick, game_result) is False


def test_grade_game_pick_no_loss():
    game_pick = {"team": "UConn", "side": "NO"}
    game_result = {
        "home_name": "UConn Huskies",
        "away_name": "Providence Friars",
        "home_score": 75,
        "away_score": 60,
    }
    assert grade_game_pick(game_pick, game_result) is False

