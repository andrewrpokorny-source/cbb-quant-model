"""Freeze the MLB training data snapshot used by the auto-research loop.

Copies `mlb_training_data_processed.csv` from repo root into
`mlb_research/anchor/mlb_frozen.csv` and records a manifest with row count,
SHA256, date range, and the optimizer/monitor window boundaries that all
experiments must share.

Idempotent: refuses to overwrite an existing frozen CSV unless `--force` is
passed. Makes the frozen CSV read-only (chmod 444) after write so an
experiment cannot accidentally mutate it.

Re-freezing is a deliberate action -- rerun with --force when we want a
fresh monitor_2026 window (the season is still accumulating).
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone

import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_CSV = os.path.join(REPO_ROOT, "mlb_training_data_processed.csv")
ANCHOR_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_CSV = os.path.join(ANCHOR_DIR, "mlb_frozen.csv")
MANIFEST_PATH = os.path.join(ANCHOR_DIR, "anchor_manifest.json")

# Fixed eval windows -- these must never change once the loop has started.
# Re-running snapshot_data.py only extends monitor_2026 forward in time
# (by picking up new games in the source CSV).
OPTIMIZER_START = "2025-04-01"
OPTIMIZER_END = "2025-07-31"
MONITOR_2025_TAIL_START = "2025-08-01"
MONITOR_2025_TAIL_END = "2025-10-31"
MONITOR_2026_START = "2026-03-25"


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore_writable(path):
    """Restore write permission so a --force snapshot can replace the file."""
    if os.path.exists(path):
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def make_read_only(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing frozen snapshot + manifest.",
    )
    args = parser.parse_args()

    if not os.path.exists(SOURCE_CSV):
        sys.exit(f"Source CSV not found: {SOURCE_CSV}")

    if os.path.exists(FROZEN_CSV) and not args.force:
        sys.exit(
            f"Frozen snapshot already exists at {FROZEN_CSV}. "
            "Pass --force to replace it."
        )

    restore_writable(FROZEN_CSV)
    shutil.copy2(SOURCE_CSV, FROZEN_CSV)

    df = pd.read_csv(FROZEN_CSV, low_memory=False)
    if "date" not in df.columns:
        sys.exit("Frozen CSV missing 'date' column.")

    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    date_min = dates.min().strftime("%Y-%m-%d")
    date_max = dates.max().strftime("%Y-%m-%d")

    def window_count(start, end=None):
        mask = dates >= start
        if end is not None:
            mask &= dates <= end
        return int(mask.sum())

    manifest = {
        "snapshot_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_csv": os.path.relpath(SOURCE_CSV, REPO_ROOT),
        "frozen_csv": os.path.relpath(FROZEN_CSV, REPO_ROOT),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "sha256": sha256_of_file(FROZEN_CSV),
        "date_min": date_min,
        "date_max": date_max,
        "windows": {
            "optimizer": {
                "start": OPTIMIZER_START,
                "end": OPTIMIZER_END,
                "row_count": window_count(OPTIMIZER_START, OPTIMIZER_END),
            },
            "monitor_2025_tail": {
                "start": MONITOR_2025_TAIL_START,
                "end": MONITOR_2025_TAIL_END,
                "row_count": window_count(MONITOR_2025_TAIL_START, MONITOR_2025_TAIL_END),
            },
            "monitor_2026": {
                "start": MONITOR_2026_START,
                "end": date_max,
                "row_count": window_count(MONITOR_2026_START),
            },
        },
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    make_read_only(FROZEN_CSV)

    print(f"Frozen snapshot written to {FROZEN_CSV}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
