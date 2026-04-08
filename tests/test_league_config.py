"""Tests for shared league config helpers."""

import pytest

from league_config import (
    get_league_artifact_paths,
    get_season_start_date,
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
    assert paths["model_file"].endswith("models/womens_cbb_spread_model_v2.pkl")
    assert paths["win_model_file"].endswith("models/womens_cbb_win_model_v1.pkl")
    assert paths["odds_archive_file"].endswith("data/odds_history.csv")
    assert paths["predictions_archive_prefix"] == "data/predictions_wbb"
    assert paths["torvik_snapshot_file"] is None
    assert paths["womens_net_snapshot_file"].endswith("data/womens_net_snapshots.csv")
    assert paths["womens_net_map_file"].endswith("data/womens_net_team_map.csv")


def test_artifact_paths_include_torvik_files_for_mens(tmp_path):
    paths = get_league_artifact_paths(str(tmp_path), "mens")
    assert paths["torvik_snapshot_file"].endswith("data/torvik_ratings_snapshots.csv")
    assert paths["torvik_map_file"].endswith("data/torvik_team_map.csv")
    assert paths["hasla_snapshot_file"].endswith("data/hasla_rank_snapshots.csv")
    assert paths["hasla_map_file"].endswith("data/hasla_team_map.csv")


def test_scoreboard_base_url_uses_league_path():
    mens_url = get_scoreboard_base_url("mens")
    womens_url = get_scoreboard_base_url("womens")
    assert "mens-college-basketball" in mens_url
    assert "womens-college-basketball" in womens_url


def test_season_start_date_is_configured_per_league():
    assert get_season_start_date("mens") == "2025-11-04"
    assert get_season_start_date("womens") == "2025-11-05"
