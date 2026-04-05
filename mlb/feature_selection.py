"""MLB feature selection: permutation importance + backtest comparison."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import get_feature_list, load_model, train_and_evaluate, TARGET_BY_LEAGUE
from league_config import get_league_artifact_paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAGUE = "mlb"


def compute_permutation_importance(X, y, model=None, n_repeats=10):
    """Compute permutation importance for each feature.

    If no model is provided, trains a fresh one.
    Returns dict of {feature_name: mean_importance}.
    """
    if model is None:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(X, y)

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, scoring="accuracy",
    )
    return {feat: imp for feat, imp in zip(X.columns, result.importances_mean)}


def prune_features(importance_dict, threshold=0.0):
    """Return list of features with importance above threshold."""
    return [f for f, imp in importance_dict.items() if imp > threshold]


def run_selection(candidate_features=None):
    """Run feature selection pipeline on MLB data.

    1. Load processed data
    2. Train with all candidate features
    3. Compute permutation importance on holdout
    4. Print importance ranking
    5. Return recommended feature list
    """
    paths = get_league_artifact_paths(BASE_DIR, LEAGUE)
    data_file = paths["data_file"]
    target = TARGET_BY_LEAGUE[LEAGUE]

    if candidate_features is None:
        candidate_features = get_feature_list(LEAGUE)

    print(f"=== MLB FEATURE SELECTION ===")
    print(f"Candidates: {len(candidate_features)} features")

    df = pd.read_csv(data_file, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    # Home rows only (matches betting evaluation)
    home = df[df["is_home"] == 1].copy()
    clean = home.dropna(subset=candidate_features + [target])
    print(f"Clean rows: {len(clean)} (from {len(home)} home games)")

    if len(clean) < 100:
        print("Not enough clean data for feature selection.")
        return candidate_features

    # Chronological 80/20 split
    clean = clean.sort_values("date").reset_index(drop=True)
    split_idx = int(len(clean) * 0.8)
    train_df = clean.iloc[:split_idx]
    test_df = clean.iloc[split_idx:]

    X_train = train_df[candidate_features]
    y_train = train_df[target]
    X_test = test_df[candidate_features]
    y_test = test_df[target]

    # Train
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    base_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    base_model.fit(X_train, y_train)

    # Accuracy on holdout
    acc = base_model.score(X_test, y_test)
    print(f"\nHoldout accuracy (all {len(candidate_features)} features): {acc:.1%}")

    # Permutation importance
    print(f"\nPermutation importance (n_repeats=10):\n")
    importance = compute_permutation_importance(X_test, y_test, model=base_model)

    # Sort by importance descending
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_imp:
        marker = " ***" if imp <= 0 else ""
        print(f"  {feat:35s} {imp:+.4f}{marker}")

    # Prune
    selected = prune_features(importance, threshold=0.0)
    dropped = [f for f in candidate_features if f not in selected]

    print(f"\n--- RECOMMENDATION ---")
    print(f"Keep: {len(selected)} features")
    print(f"Drop: {len(dropped)} features")
    if dropped:
        print(f"  Dropped: {', '.join(dropped)}")

    # Compare accuracy with pruned set
    if len(selected) < len(candidate_features) and len(selected) > 0:
        X_train_pruned = train_df[selected]
        X_test_pruned = test_df[selected]
        pruned_model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        pruned_model.fit(X_train_pruned, y_train)
        pruned_acc = pruned_model.score(X_test_pruned, y_test)
        print(f"\nHoldout accuracy (pruned {len(selected)} features): {pruned_acc:.1%}")
        print(f"Change: {pruned_acc - acc:+.1%}")

    return selected


def main():
    parser = argparse.ArgumentParser(description="MLB feature selection")
    args = parser.parse_args()
    selected = run_selection()
    print(f"\nFinal feature list ({len(selected)}):")
    for f in selected:
        print(f"    '{f}',")


if __name__ == "__main__":
    main()
