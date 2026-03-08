"""Helpers for storing append-only market line snapshots."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd


ARCHIVE_COLUMNS = [
    "captured_at",
    "league",
    "date",
    "home_team",
    "away_team",
    "spread",
    "total_line",
    "book",
    "provider",
    "source",
    "raw_line",
    "has_market_spread",
]

DEFAULT_ARCHIVE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "odds_history.csv",
)


def infer_line_source(spread_source: str | None) -> tuple[str, str]:
    """Map prediction-time spread text into stable archive metadata."""
    source_text = str(spread_source or "").strip()
    lower = source_text.lower()

    if lower.startswith("manual "):
        return "Manual", "manual_override"
    if lower.startswith("kalshi "):
        return "Kalshi", "kalshi_market"
    if source_text and source_text != "0":
        return "ESPN", "espn_scoreboard"
    return "", "missing"


def build_archive_record(
    *,
    league: str,
    game_date,
    home_team: str,
    away_team: str,
    spread,
    spread_source: str | None,
    raw_line: str | None = None,
    total_line=None,
    captured_at: datetime | None = None,
):
    """Build one normalized archive row from a resolved game line."""
    ts = captured_at or datetime.now(timezone.utc)
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    book, provider = infer_line_source(spread_source)
    date_value = pd.to_datetime(game_date).strftime("%Y-%m-%d")
    has_market_spread = spread is not None and not pd.isna(spread)

    return {
        "captured_at": ts.astimezone(timezone.utc).isoformat(),
        "league": str(league),
        "date": date_value,
        "home_team": str(home_team),
        "away_team": str(away_team),
        "spread": float(spread) if has_market_spread else pd.NA,
        "total_line": float(total_line) if total_line is not None and not pd.isna(total_line) else pd.NA,
        "book": book,
        "provider": provider,
        "source": "prediction_run",
        "raw_line": str(raw_line or spread_source or ""),
        "has_market_spread": bool(has_market_spread),
    }


def append_archive_records(records, archive_file: str = DEFAULT_ARCHIVE_FILE) -> int:
    """Append non-duplicate archive rows to the local odds history CSV."""
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
    else:
        combined = incoming

    before = 0
    if os.path.exists(archive_file):
        before = len(pd.read_csv(archive_file))
    combined.to_csv(archive_file, index=False)
    return len(combined) - before
