import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from scipy.stats import norm

from league_config import get_league_artifact_paths, normalize_league

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
    'is_home',
    'spread',
    'rest_days',
    'diff_eFG',
    'diff_Rebound',
    'diff_TO',
    'momentum_gap',
    'roll5_cover_margin',
    'prev_games_played',
    'opp_win_pct',
    'prev_blowout_rate',
    'prev_roll5_margin',
    'prev_volatility',
    'spread_abs',
    'spread_squared',
]

def train_and_evaluate(league="mens"):
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    model_file = paths["model_file"]

    print(f"--- TRAINING CBB MODEL ({league}, GBM + Sigmoid Calibration, 15 features) ---")

    if not os.path.exists(data_file):
        print("No processed data found. Run features.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows.")

    # Compute derived spread features
    df['spread_abs'] = df['spread'].abs()
    df['spread_squared'] = df['spread'] ** 2

    # 2. Define Features
    features = FEATURES
    target = 'ats_win'

    # 3. Validation: Ensure all columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"CRITICAL ERROR: Missing features in CSV: {missing_cols}")
        print("Run 'python3 features.py' again to regenerate them.")
        return

    # 4. Clean & Prep
    df_model = df.dropna(subset=features + [target]).copy()

    # Force float types for inputs
    X = df_model[features].astype(float)
    y = df_model[target].astype(int)

    print(f"Training on {len(X)} clean games.")
    print(f"Features: {len(features)}")

    # 5. Split data: train / test (time-series aware)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 6. Train base model (uncalibrated) for comparison
    base_clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    base_clf.fit(X_train, y_train)

    # 7. Evaluate UNCALIBRATED model
    uncal_probs = base_clf.predict_proba(X_test)[:, 1]
    uncal_preds = base_clf.predict(X_test)
    uncal_acc = accuracy_score(y_test, uncal_preds)
    uncal_brier = brier_score_loss(y_test, uncal_probs)

    # 8. Train CALIBRATED model using cross-validation
    # CalibratedClassifierCV with cv=5 will handle calibration internally
    calibrated_clf = CalibratedClassifierCV(
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),
        method='sigmoid',
        cv=5
    )
    calibrated_clf.fit(X_train, y_train)

    # 9. Evaluate CALIBRATED model
    cal_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    cal_preds = calibrated_clf.predict(X_test)
    cal_acc = accuracy_score(y_test, cal_preds)
    cal_brier = brier_score_loss(y_test, cal_probs)

    print(f"\n=== CALIBRATION COMPARISON ===")
    print(f"{'Metric':<20} {'Uncalibrated':<15} {'Calibrated':<15} {'Change':<10}")
    print(f"{'-'*60}")
    print(f"{'Accuracy':<20} {uncal_acc:<15.1%} {cal_acc:<15.1%} {(cal_acc-uncal_acc)*100:+.1f}%")
    print(f"{'Brier Score':<20} {uncal_brier:<15.4f} {cal_brier:<15.4f} {cal_brier-uncal_brier:+.4f}")

    # 10. Calibration analysis - compare predicted vs actual by confidence bucket
    print(f"\n=== CALIBRATION BY CONFIDENCE BUCKET ===")
    print(f"{'Predicted':<15} {'Actual (uncal)':<18} {'Actual (cal)':<15}")

    for low, high in [(0.50, 0.53), (0.53, 0.55), (0.55, 0.57), (0.57, 0.60), (0.60, 0.65)]:
        # Uncalibrated
        uncal_mask = (uncal_probs >= low) & (uncal_probs < high)
        if uncal_mask.sum() > 0:
            uncal_actual = y_test[uncal_mask].mean()
        else:
            uncal_actual = 0

        # Calibrated
        cal_mask = (cal_probs >= low) & (cal_probs < high)
        if cal_mask.sum() > 0:
            cal_actual = y_test[cal_mask].mean()
        else:
            cal_actual = 0

        mid = (low + high) / 2
        print(f"{mid:.1%}             {uncal_actual:.1%} (n={uncal_mask.sum():<4})    {cal_actual:.1%} (n={cal_mask.sum()})")

    # 11. High confidence accuracy
    high_conf_mask = (cal_probs > 0.53) | (cal_probs < 0.47)
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(y_test[high_conf_mask], cal_preds[high_conf_mask])
        high_conf_count = high_conf_mask.sum()
    else:
        high_conf_acc = 0
        high_conf_count = 0

    print(f"\nHigh Confidence (>53%): {high_conf_acc:.1%} ({high_conf_count} bets)")

    # 12. Feature importance (from base model)
    print(f"\n=== TOP FEATURES ===")
    importance = pd.DataFrame({
        'feature': features,
        'importance': base_clf.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:<20}: {row['importance']:.3f}")

    # 13. Compute sigma for CDF-based line shopping
    # sigma = std(actual_margin - vegas_predicted_margin) where vegas margin = -spread
    # This measures how much actual outcomes deviate from the spread, used to
    # project the classifier's probability at the market spread to other spreads
    # via norm.cdf curves.
    if 'margin' in df_model.columns:
        train_indices = X_train.index
        train_margins = df_model.loc[train_indices, 'margin'].values
        train_spreads = df_model.loc[train_indices, 'spread'].values
        residuals = train_margins - (-train_spreads)  # actual - vegas prediction
        sigma = float(np.std(residuals))
        print(f"\nLine shopping sigma: {sigma:.2f} (std of margin vs spread)")
    else:
        sigma = 11.0  # Reasonable CBB default
        print(f"\nLine shopping sigma: {sigma:.2f} (default, margin column not found)")

    # 14. Save calibrated model + sigma
    joblib.dump({'model': calibrated_clf, 'sigma': sigma}, model_file)
    print(f"Model + sigma saved to {model_file}")


def load_model(path=None, league="mens"):
    """
    Load classifier + sigma from pkl file.

    Handles both old format (raw model) and new format ({'model': ..., 'sigma': ...}).
    Returns (model, sigma).
    """
    if path is None:
        league = normalize_league(league)
        path = get_league_artifact_paths(BASE_DIR, league)["model_file"]

    data = joblib.load(path)
    if isinstance(data, dict) and 'model' in data:
        sigma = data.get('sigma', 11.0)
        return data['model'], float(sigma)
    # Old format: raw model without sigma
    return data, 11.0


def cover_prob_at_spread(classifier_prob, market_spread, alt_spread, sigma):
    """
    Project a classifier's cover probability to a different spread using CDF.

    The classifier gives P(home covers) at the market spread. To get the
    probability at a different spread, we:
    1. Derive an "effective margin" that would produce the classifier's prob
    2. Evaluate the CDF at the alternative spread using that margin

    This guarantees:
    - At market_spread: returns exactly classifier_prob
    - Monotonic: probability increases as spread becomes more favorable
    - Smooth: normal CDF shape between spreads

    Args:
        classifier_prob: P(home covers) from the classifier at market spread
        market_spread: The spread the classifier was evaluated at
        alt_spread: The spread to project to
        sigma: Std dev of (actual_margin - (-spread)) from training data

    Returns:
        P(home covers) at alt_spread
    """
    # Clamp to avoid inf from norm.ppf at 0 or 1
    p = np.clip(classifier_prob, 0.001, 0.999)
    effective_margin = sigma * norm.ppf(p) - market_spread
    return float(norm.cdf((effective_margin + alt_spread) / sigma))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CBB spread model.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League to train: mens or womens (aliases supported).",
    )
    args = parser.parse_args()
    train_and_evaluate(args.league)
