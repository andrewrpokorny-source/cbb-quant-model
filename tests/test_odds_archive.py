from datetime import datetime, timezone

import pandas as pd

from odds_archive import append_archive_records, build_archive_record, infer_line_source


def test_infer_line_source_maps_known_sources():
    assert infer_line_source("Manual -3.5") == ("Manual", "manual_override")
    assert infer_line_source("Kalshi +4.5") == ("Kalshi", "kalshi_market")
    assert infer_line_source("DUKE -5.5") == ("ESPN", "espn_scoreboard")
    assert infer_line_source(None) == ("", "missing")


def test_build_archive_record_handles_missing_spread():
    record = build_archive_record(
        league="womens",
        game_date="2026-03-06T19:00:00Z",
        home_team="Home",
        away_team="Away",
        spread=None,
        spread_source=None,
        captured_at=datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc),
    )
    assert record["provider"] == "missing"
    assert record["has_market_spread"] is False
    assert pd.isna(record["spread"])


def test_append_archive_records_dedupes_exact_duplicates(tmp_path):
    archive_file = tmp_path / "odds_history.csv"
    record = build_archive_record(
        league="mens",
        game_date="2026-03-06",
        home_team="Duke",
        away_team="UNC",
        spread=-4.5,
        spread_source="DUKE -4.5",
        captured_at=datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc),
    )

    assert append_archive_records([record], archive_file=str(archive_file)) == 1
    assert append_archive_records([record], archive_file=str(archive_file)) == 0

    saved = pd.read_csv(archive_file)
    assert len(saved) == 1
    assert saved.iloc[0]["book"] == "ESPN"
    assert saved.iloc[0]["provider"] == "espn_scoreboard"
