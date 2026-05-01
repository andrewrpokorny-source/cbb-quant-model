"""Build or diagnose a market-aware MLB research anchor.

This is the Track 2 entry point. It deliberately refuses to freeze an anchor
unless the source CSV has enough paired moneyline coverage to support
market-implied features and moneyline ROI evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "mlb_training_data_processed.csv"
DEFAULT_OUTPUT = REPO_ROOT / "mlb_research" / "anchor_v2" / "mlb_market_frozen.csv"
DEFAULT_MANIFEST = REPO_ROOT / "mlb_research" / "anchor_v2" / "market_anchor_manifest.json"

sys.path.insert(0, str(REPO_ROOT))
from mlb_research.market_odds import (  # noqa: E402
    MARKET_FEATURE_COLUMNS,
    add_market_odds_features,
    market_coverage_summary,
)

OPTIMIZER_START = "2025-04-01"
OPTIMIZER_END = "2025-07-31"
MONITOR_2025_TAIL_START = "2025-08-01"
MONITOR_2025_TAIL_END = "2025-10-31"
MONITOR_2026_START = "2026-03-25"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def restore_writable(path: Path) -> None:
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def _window_count(dates: pd.Series, start: str, end: str | None = None) -> int:
    mask = dates >= start
    if end is not None:
        mask &= dates <= end
    return int(mask.sum())


def build_manifest(source: Path, output: Path, df: pd.DataFrame, coverage: dict) -> dict:
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    date_min = dates.min().strftime("%Y-%m-%d") if not dates.empty else ""
    date_max = dates.max().strftime("%Y-%m-%d") if not dates.empty else ""
    return {
        "snapshot_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_csv": str(source.relative_to(REPO_ROOT)),
        "frozen_csv": str(output.relative_to(REPO_ROOT)),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "sha256": sha256_of_file(output) if output.exists() else None,
        "date_min": date_min,
        "date_max": date_max,
        "market_feature_columns": MARKET_FEATURE_COLUMNS,
        "market_coverage": coverage,
        "windows": {
            "optimizer": {
                "start": OPTIMIZER_START,
                "end": OPTIMIZER_END,
                "row_count": _window_count(dates, OPTIMIZER_START, OPTIMIZER_END),
            },
            "monitor_2025_tail": {
                "start": MONITOR_2025_TAIL_START,
                "end": MONITOR_2025_TAIL_END,
                "row_count": _window_count(dates, MONITOR_2025_TAIL_START, MONITOR_2025_TAIL_END),
            },
            "monitor_2026": {
                "start": MONITOR_2026_START,
                "end": date_max,
                "row_count": _window_count(dates, MONITOR_2026_START),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source CSV not found: {args.source}")
    if args.output.exists() and not args.force and not args.diagnose_only:
        sys.exit(f"Output already exists: {args.output}. Pass --force to replace it.")

    df = pd.read_csv(args.source, low_memory=False)
    enriched = add_market_odds_features(df)
    coverage = market_coverage_summary(df)
    manifest = build_manifest(args.source.resolve(), args.output.resolve(), enriched, coverage)

    if args.diagnose_only:
        print(json.dumps(manifest, indent=2))
        return

    if coverage["complete_no_vig_share"] < args.min_coverage:
        sys.exit(
            "Refusing to freeze market anchor: paired moneyline coverage is "
            f"{coverage['complete_no_vig_share']:.1%}, below required "
            f"{args.min_coverage:.1%}. Run with --diagnose-only for details."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    restore_writable(args.output)
    enriched.to_csv(args.output, index=False)
    manifest["sha256"] = sha256_of_file(args.output)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    make_read_only(args.output)

    print(f"Market anchor written to {args.output}")
    print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
