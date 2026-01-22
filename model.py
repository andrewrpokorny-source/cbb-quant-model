import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
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
    print("--- 🤖 TRAINING CBB MODEL V2 (Gradient Boosting) 🤖 ---")

    if not os.path.exists(DATA_FILE):
        print("❌ No processed data found. Run features.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    print(f"   -> Loaded {len(df)} rows.")

    # 2. Define Features
    features = FEATURES
    target = 'ats_win'

    # 3. Validation: Ensure all columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"❌ CRITICAL ERROR: Missing features in CSV: {missing_cols}")
        print("   -> Run 'python3 features.py' again to regenerate them.")
        return

    # 4. Clean & Prep
    df_model = df.dropna(subset=features + [target]).copy()

    # Force float types for inputs
    X = df_model[features].astype(float)
    y = df_model[target].astype(int)

    print(f"   -> Training on {len(X)} clean games.")
    print(f"   -> Features: {len(features)}")

    # 5. Split & Train (time-series aware)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Gradient Boosting (V2 - better than RF in experiments)
    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # 6. Evaluate
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)

    # High confidence accuracy (actionable bets)
    high_conf_mask = (probs > 0.53) | (probs < 0.47)
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(y_test[high_conf_mask], preds[high_conf_mask])
        high_conf_count = high_conf_mask.sum()
    else:
        high_conf_acc = 0
        high_conf_count = 0

    print(f"\n   🎯 Validation Results:")
    print(f"      Overall Accuracy:    {acc:.1%}")
    print(f"      Brier Score:         {brier:.4f}")
    print(f"      High Conf Accuracy:  {high_conf_acc:.1%} ({high_conf_count} bets)")

    # Feature importance
    print(f"\n   📊 Top 5 Features:")
    importance = pd.DataFrame({
        'feature': features,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(5).iterrows():
        print(f"      {row['feature']:<20}: {row['importance']:.3f}")

    # 7. Save
    joblib.dump(clf, MODEL_FILE)
    print(f"\n✅ Model Saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_evaluate()