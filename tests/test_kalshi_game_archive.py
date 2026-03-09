"""Tests for Kalshi GAME snapshot archive helpers."""

from datetime import datetime, timezone

import pandas as pd

from kalshi_game_archive import append_archive_records, build_game_archive_record


def test_build_game_archive_record_normalizes_fields():
    record = build_game_archive_record(
        league="mens",
        game_datetime=datetime(2026, 3, 8, 19, 0, tzinfo=timezone.utc),
        home_team="UConn Huskies",
        away_team="Providence Friars",
        matchup="Providence Friars @ UConn Huskies",
        pick="UConn Huskies ML YES",
        picked_team="UConn Huskies",
        kalshi_side="YES",
        kalshi_ticker="KXNCAAMBGAME-EXAMPLE",
        kalshi_title="Resolves to YES if UConn wins",
        kalshi_yes_team="UConn Huskies",
        kalshi_yes_price=58,
        kalshi_no_price=44,
        kalshi_price=58,
        kalshi_fee=1.7,
        win_model_home_prob=0.61,
        conf=0.61,
        edge=0.023,
        edge_pct=2.3,
        rating="MARGINAL",
        units=0.5,
        captured_at=datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc),
    )

    assert record["league"] == "mens"
    assert record["game_date"] == "2026-03-08"
    assert record["kalshi_yes_price"] == 58.0
    assert record["edge_pct"] == 2.3
    assert record["captured_at"].endswith("+00:00")


def test_append_archive_records_deduplicates(tmp_path):
    archive_file = tmp_path / "kalshi_game_history.csv"
    record = build_game_archive_record(
        league="mens",
        game_datetime=datetime(2026, 3, 8, 19, 0, tzinfo=timezone.utc),
        home_team="UConn Huskies",
        away_team="Providence Friars",
        matchup="Providence Friars @ UConn Huskies",
        pick="UConn Huskies ML YES",
        picked_team="UConn Huskies",
        kalshi_side="YES",
        kalshi_ticker="KXNCAAMBGAME-EXAMPLE",
        kalshi_title="Resolves to YES if UConn wins",
        kalshi_yes_team="UConn Huskies",
        kalshi_yes_price=58,
        kalshi_no_price=44,
        kalshi_price=58,
        kalshi_fee=1.7,
        win_model_home_prob=0.61,
        conf=0.61,
        edge=0.023,
        edge_pct=2.3,
        rating="MARGINAL",
        units=0.5,
        captured_at=datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc),
    )

    assert append_archive_records([record], str(archive_file)) == 1
    assert append_archive_records([record], str(archive_file)) == 0

    df = pd.read_csv(archive_file)
    assert len(df) == 1
    assert df.loc[0, "kalshi_ticker"] == "KXNCAAMBGAME-EXAMPLE"
