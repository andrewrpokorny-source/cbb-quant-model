"""One-time migration: move root-level generated files into data/ and models/.

Run after pulling the repo reorganization commit. Safe to run multiple times;
it only moves files that still exist at the old location.
"""

import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVES = [
    # (old_path_relative, new_path_relative)
    # Generated CSVs -> data/
    ("betting_history.csv", "data/betting_history.csv"),
    ("daily_predictions.csv", "data/daily_predictions.csv"),
    ("daily_predictions_wbb.csv", "data/daily_predictions_wbb.csv"),
    ("daily_predictions_mlb.csv", "data/daily_predictions_mlb.csv"),
    ("performance_log.csv", "data/performance_log.csv"),
    ("performance_log_wbb.csv", "data/performance_log_wbb.csv"),
    ("performance_log_mlb.csv", "data/performance_log_mlb.csv"),
    ("odds_history.csv", "data/odds_history.csv"),
    ("kalshi_game_history.csv", "data/kalshi_game_history.csv"),
    ("draft_results.csv", "data/draft_results.csv"),
    # Generated PKL -> models/
    ("wbb_model_v2.pkl", "models/wbb_model_v2.pkl"),
]


def migrate():
    moved = []
    skipped = []

    for old_rel, new_rel in MOVES:
        old = os.path.join(BASE_DIR, old_rel)
        new = os.path.join(BASE_DIR, new_rel)
        if not os.path.exists(old):
            continue
        if os.path.exists(new):
            skipped.append((old_rel, "already exists at new location"))
            continue
        shutil.move(old, new)
        moved.append((old_rel, new_rel))

    # Also migrate any prediction archives (predictions_*.csv, predictions_wbb_*.csv)
    import glob
    for pattern in ("predictions_*.csv", "predictions_wbb_*.csv", "predictions_mlb_*.csv"):
        for old in glob.glob(os.path.join(BASE_DIR, pattern)):
            basename = os.path.basename(old)
            new = os.path.join(BASE_DIR, "data", basename)
            if os.path.exists(new):
                skipped.append((basename, "already exists at new location"))
                continue
            shutil.move(old, new)
            moved.append((basename, f"data/{basename}"))

    if moved:
        print(f"Migrated {len(moved)} file(s):")
        for old_rel, new_rel in moved:
            print(f"  {old_rel} -> {new_rel}")
    else:
        print("Nothing to migrate.")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    migrate()
