from datetime import datetime

import pandas as pd
import pytest

from main import (
    _build_rate_snapshot,
    backfill_venue_metadata,
    ensure_venue_columns,
    ensure_raw_rate_columns,
    get_last_recorded_date,
    merge_raw_rate_data,
    merge_venue_metadata,
)


def test_ensure_venue_columns_adds_defaults():
    df = pd.DataFrame({"date": ["2026-01-01"], "team": ["A"]})
    result = ensure_venue_columns(df)
    assert list(result["is_neutral"]) == [0]
    assert list(result["venue_city"]) == [""]
    assert list(result["venue_state"]) == [""]


def test_merge_venue_metadata_backfills_blank_rows():
    existing = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01"],
            "team": ["Home", "Away"],
            "is_neutral": [0, 0],
            "venue_city": ["", ""],
            "venue_state": ["", ""],
        }
    )
    metadata = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01"],
            "team": ["Home", "Away"],
            "is_neutral": [1, 1],
            "venue_city": ["Charlotte", "Charlotte"],
            "venue_state": ["NC", "NC"],
        }
    )
    result = merge_venue_metadata(existing, metadata)
    assert set(result["venue_city"]) == {"Charlotte"}
    assert set(result["venue_state"]) == {"NC"}
    assert set(result["is_neutral"]) == {1}


def test_merge_venue_metadata_preserves_existing_values():
    existing = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "team": ["Home"],
            "is_neutral": [1],
            "venue_city": ["Las Vegas"],
            "venue_state": ["NV"],
        }
    )
    metadata = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "team": ["Home"],
            "is_neutral": [0],
            "venue_city": ["Charlotte"],
            "venue_state": ["NC"],
        }
    )
    result = merge_venue_metadata(existing, metadata)
    row = result.iloc[0]
    assert row["is_neutral"] == 1
    assert row["venue_city"] == "Las Vegas"
    assert row["venue_state"] == "NV"


def test_backfill_venue_metadata_fetches_missing_dates(monkeypatch):
    df = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "team": ["Home", "Away"],
            "is_neutral": [0, 0],
            "venue_city": ["", ""],
            "venue_state": ["", ""],
        }
    )

    def fake_fetch(target_date, _base_url):
        ds = target_date.strftime("%Y-%m-%d")
        return [
            {
                "date": ds,
                "team": "Home" if ds == "2026-01-01" else "Away",
                "opponent": "X",
                "is_home": 1,
                "spread": 0.0,
                "team_score": 0,
                "opp_score": 0,
                "location": "Home",
                "ats_win": 0,
                "is_neutral": 0,
                "venue_city": "Piscataway" if ds == "2026-01-01" else "Storrs",
                "venue_state": "NJ" if ds == "2026-01-01" else "CT",
            }
        ]

    monkeypatch.setattr("main.fetch_games_for_date", fake_fetch)
    result = backfill_venue_metadata(df, "http://example.com")
    assert set(result["venue_city"]) == {"Piscataway", "Storrs"}
    assert set(result["venue_state"]) == {"NJ", "CT"}


def test_get_last_recorded_date_uses_configured_season_start_when_missing(tmp_path):
    season_start = datetime(2025, 11, 4)
    result = get_last_recorded_date(tmp_path / "missing.csv", season_start)
    assert result == season_start


def test_ensure_raw_rate_columns_adds_defaults():
    df = pd.DataFrame({"date": ["2026-01-01"], "team": ["A"]})
    result = ensure_raw_rate_columns(df)
    assert "possessions" in result.columns
    assert "team_eFG" in result.columns
    assert "opp_3PR" in result.columns


def test_merge_raw_rate_data_backfills_missing_rate_fields():
    existing = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "team": ["Home"],
            "opponent": ["Away"],
            "is_home": [1],
            "possessions": [pd.NA],
            "team_eFG": [pd.NA],
            "team_TO": [pd.NA],
            "team_ORB": [pd.NA],
            "team_FTR": [pd.NA],
            "team_3PR": [pd.NA],
            "opp_eFG": [pd.NA],
            "opp_TO": [pd.NA],
            "opp_ORB": [pd.NA],
            "opp_FTR": [pd.NA],
            "opp_3PR": [pd.NA],
        }
    )
    rates = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "team": ["Home"],
            "opponent": ["Away"],
            "is_home": [1],
            "possessions": [70.0],
            "team_eFG": [55.0],
            "team_TO": [16.0],
            "team_ORB": [31.0],
            "team_FTR": [28.0],
            "team_3PR": [37.0],
            "opp_eFG": [48.0],
            "opp_TO": [18.0],
            "opp_ORB": [24.0],
            "opp_FTR": [22.0],
            "opp_3PR": [33.0],
        }
    )

    result = merge_raw_rate_data(existing, rates)
    row = result.iloc[0]
    assert row["possessions"] == 70.0
    assert row["team_eFG"] == 55.0
    assert row["opp_3PR"] == 33.0


def test_build_rate_snapshot_matches_expected_percentage_scales():
    team_stats = {
        "fieldGoalsMade-fieldGoalsAttempted": "24-53",
        "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "9-24",
        "freeThrowsMade-freeThrowsAttempted": "4-4",
        "offensiveRebounds": "5",
        "defensiveRebounds": "24",
        "totalTurnovers": "14",
    }
    opp_stats = {
        "fieldGoalsMade-fieldGoalsAttempted": "29-69",
        "threePointFieldGoalsMade-threePointFieldGoalsAttempted": "10-32",
        "freeThrowsMade-freeThrowsAttempted": "8-15",
        "offensiveRebounds": "18",
        "defensiveRebounds": "24",
        "totalTurnovers": "8",
    }

    result = _build_rate_snapshot(team_stats, opp_stats)

    assert result["team_eFG"] == pytest.approx(53.7736, rel=1e-4)
    assert result["team_3PR"] == pytest.approx(45.2830, rel=1e-4)
    assert result["team_FTR"] == pytest.approx(7.5472, rel=1e-4)
    assert result["team_TO"] == pytest.approx(21.9092, rel=1e-4)
    assert result["team_ORB"] == pytest.approx(17.2414, rel=1e-4)
    assert result["opp_ORB"] == pytest.approx(42.8571, rel=1e-4)
