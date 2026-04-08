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


def _pick_and_move(old, new, old_rel, new_rel, moved, skipped):
    """Move old -> new. If both exist, keep the larger file (the real data)."""
    if not os.path.exists(old):
        return
    if os.path.exists(new):
        old_size = os.path.getsize(old)
        new_size = os.path.getsize(new)
        if old_size > new_size:
            os.remove(new)
            shutil.move(old, new)
            moved.append((old_rel, new_rel, f"replaced smaller destination ({new_size}B < {old_size}B)"))
        elif old_size == new_size:
            os.remove(old)
            skipped.append((old_rel, "identical size at both locations, removed old"))
        else:
            os.remove(old)
            skipped.append((old_rel, f"destination is larger ({new_size}B > {old_size}B), removed old"))
    else:
        shutil.move(old, new)
        moved.append((old_rel, new_rel, None))


def migrate():
    moved = []
    skipped = []

    for old_rel, new_rel in MOVES:
        old = os.path.join(BASE_DIR, old_rel)
        new = os.path.join(BASE_DIR, new_rel)
        _pick_and_move(old, new, old_rel, new_rel, moved, skipped)

    # Also migrate any prediction archives (predictions_*.csv, predictions_wbb_*.csv)
    import glob
    for pattern in ("predictions_*.csv", "predictions_wbb_*.csv", "predictions_mlb_*.csv"):
        for old in glob.glob(os.path.join(BASE_DIR, pattern)):
            basename = os.path.basename(old)
            new = os.path.join(BASE_DIR, "data", basename)
            _pick_and_move(old, new, basename, f"data/{basename}", moved, skipped)

    if moved:
        print(f"Migrated {len(moved)} file(s):")
        for old_rel, new_rel, note in moved:
            suffix = f" ({note})" if note else ""
            print(f"  {old_rel} -> {new_rel}{suffix}")
    else:
        print("Nothing to migrate.")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    migrate()
