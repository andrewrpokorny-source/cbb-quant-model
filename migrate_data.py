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


def _pick_and_move(old, new, old_rel, new_rel, moved, conflicts):
    """Move old -> new. If both exist, flag a conflict for manual resolution."""
    if not os.path.exists(old):
        return
    if os.path.exists(new):
        old_size = os.path.getsize(old)
        new_size = os.path.getsize(new)
        conflicts.append((old_rel, new_rel, old_size, new_size))
    else:
        shutil.move(old, new)
        moved.append((old_rel, new_rel))


def migrate():
    moved = []
    conflicts = []

    for subdir in ("data", "models"):
        os.makedirs(os.path.join(BASE_DIR, subdir), exist_ok=True)

    for old_rel, new_rel in MOVES:
        old = os.path.join(BASE_DIR, old_rel)
        new = os.path.join(BASE_DIR, new_rel)
        _pick_and_move(old, new, old_rel, new_rel, moved, conflicts)

    # Also migrate any prediction archives (predictions_*.csv, predictions_wbb_*.csv)
    import glob
    for pattern in ("predictions_*.csv", "predictions_wbb_*.csv", "predictions_mlb_*.csv"):
        for old in glob.glob(os.path.join(BASE_DIR, pattern)):
            basename = os.path.basename(old)
            new = os.path.join(BASE_DIR, "data", basename)
            _pick_and_move(old, new, basename, f"data/{basename}", moved, conflicts)

    if moved:
        print(f"Migrated {len(moved)} file(s):")
        for old_rel, new_rel in moved:
            print(f"  {old_rel} -> {new_rel}")
    else:
        print("Nothing to migrate.")

    if conflicts:
        print(f"\nCONFLICT: {len(conflicts)} file(s) exist at both old and new locations.")
        print("Resolve manually -- decide which copy has the real data, then delete the other.\n")
        for old_rel, new_rel, old_size, new_size in conflicts:
            print(f"  {old_rel} ({old_size}B)  vs  {new_rel} ({new_size}B)")
        return False

    return True


if __name__ == "__main__":
    import sys
    success = migrate()
    if not success:
        sys.exit(1)
