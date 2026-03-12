import pandas as pd

from main import backfill_venue_metadata, ensure_venue_columns, merge_venue_metadata


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
