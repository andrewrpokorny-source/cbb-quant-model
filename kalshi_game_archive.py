"""Helpers for storing append-only Kalshi GAME market snapshots."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd


ARCHIVE_COLUMNS = [
    "captured_at",
    "league",
    "game_date",
    "game_datetime",
    "matchup",
    "home_team",
    "away_team",
    "pick",
    "picked_team",
    "kalshi_side",
    "kalshi_ticker",
    "kalshi_title",
    "kalshi_yes_team",
    "kalshi_yes_price",
    "kalshi_no_price",
    "kalshi_price",
    "kalshi_fee",
    "win_model_home_prob",
    "conf",
    "edge",
    "edge_pct",
    "rating",
    "units",
    "source",
]

DEFAULT_ARCHIVE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "kalshi_game_history.csv",
)


def _isoformat_utc(ts: datetime) -> str:
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def build_game_archive_record(
    *,
    league: str,
    game_datetime,
    home_team: str,
    away_team: str,
    matchup: str,
    pick: str,
    picked_team: str | None,
    kalshi_side: str | None,
    kalshi_ticker: str | None,
    kalshi_title: str | None,
    kalshi_yes_team: str | None,
    kalshi_yes_price,
    kalshi_no_price,
    kalshi_price,
    kalshi_fee,
    win_model_home_prob,
    conf,
    edge,
    edge_pct,
    rating: str | None,
    units,
    captured_at: datetime | None = None,
    source: str = "prediction_run",
):
    """Build one normalized Kalshi GAME archive row."""
    ts = captured_at or datetime.now(timezone.utc)
    game_ts = pd.to_datetime(game_datetime)
    if getattr(game_ts, "tzinfo", None) is None:
        game_ts = game_ts.tz_localize(timezone.utc)

    def _optional_float(value):
        if value is None or pd.isna(value):
            return pd.NA
        return float(value)

    return {
        "captured_at": _isoformat_utc(ts),
        "league": str(league),
        "game_date": game_ts.strftime("%Y-%m-%d"),
        "game_datetime": _isoformat_utc(game_ts.to_pydatetime()),
        "matchup": str(matchup),
        "home_team": str(home_team),
        "away_team": str(away_team),
        "pick": str(pick),
        "picked_team": str(picked_team or ""),
        "kalshi_side": str(kalshi_side or ""),
        "kalshi_ticker": str(kalshi_ticker or ""),
        "kalshi_title": str(kalshi_title or ""),
        "kalshi_yes_team": str(kalshi_yes_team or ""),
        "kalshi_yes_price": _optional_float(kalshi_yes_price),
        "kalshi_no_price": _optional_float(kalshi_no_price),
        "kalshi_price": _optional_float(kalshi_price),
        "kalshi_fee": _optional_float(kalshi_fee),
        "win_model_home_prob": _optional_float(win_model_home_prob),
        "conf": _optional_float(conf),
        "edge": _optional_float(edge),
        "edge_pct": _optional_float(edge_pct),
        "rating": str(rating or ""),
        "units": _optional_float(units),
        "source": str(source),
    }


def append_archive_records(records, archive_file: str = DEFAULT_ARCHIVE_FILE) -> int:
    """Append non-duplicate archive rows to the local Kalshi GAME history CSV."""
    if not records:
        return 0

    incoming = pd.DataFrame(records)
    for col in ARCHIVE_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = pd.NA
    incoming = incoming[ARCHIVE_COLUMNS]
    incoming = incoming.drop_duplicates()

    if os.path.exists(archive_file):
        existing = pd.read_csv(archive_file)
        for col in ARCHIVE_COLUMNS:
            if col not in existing.columns:
                existing[col] = pd.NA
        combined = pd.concat([existing[ARCHIVE_COLUMNS], incoming], ignore_index=True)
        combined = combined.drop_duplicates()
        before = len(existing)
    else:
        combined = incoming
        before = 0

    combined.to_csv(archive_file, index=False)
    return len(combined) - before
