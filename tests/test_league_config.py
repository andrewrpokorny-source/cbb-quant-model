"""Tests for shared league config helpers."""

import pytest

from league_config import (
    get_league_artifact_paths,
    get_scoreboard_base_url,
    normalize_league,
)


def test_normalize_league_aliases():
    assert normalize_league("men") == "mens"
    assert normalize_league("wcbb") == "womens"


def test_normalize_league_invalid():
    with pytest.raises(ValueError):
        normalize_league("nba")


def test_artifact_paths_include_womens_files(tmp_path):
    paths = get_league_artifact_paths(str(tmp_path), "womens")
    assert paths["model_file"].endswith("womens_cbb_spread_model_v2.pkl")
    assert paths["win_model_file"].endswith("womens_cbb_win_model_v1.pkl")
    assert paths["predictions_archive_prefix"] == "predictions_wbb"


def test_scoreboard_base_url_uses_league_path():
    mens_url = get_scoreboard_base_url("mens")
    womens_url = get_scoreboard_base_url("womens")
    assert "mens-college-basketball" in mens_url
    assert "womens-college-basketball" in womens_url

