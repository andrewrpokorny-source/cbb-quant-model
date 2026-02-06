import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
MODEL_FILE = os.path.join(BASE_DIR, "cbb_model_v1.pkl")

# V2 Feature Set (improved from experiments)
FEATURES = [
    # Original features
    'is_home',
    'spread',
    'rest_days',
    'diff_eFG',
    'diff_Rebound',
    'diff_TO',
    'momentum_gap',
    'roll5_cover_margin',
    # New V2 features
    'prev_games_played',
    'opp_win_pct',
    'prev_blowout_rate',
    'prev_roll5_margin',
    'prev_volatility',
]

def train_and_evaluate():
    print("--- TRAINING CBB MODEL V3 (Gradient Boosting + Calibration) ---")

    if not os.path.exists(DATA_FILE):
        print("No processed data found. Run features.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows.")

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
        max_depth=3,
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
            max_depth=3,
            random_state=42
        ),
        method='isotonic',
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

    # 13. Save CALIBRATED model
    joblib.dump(calibrated_clf, MODEL_FILE)
    print(f"\nCalibrated model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_evaluate()