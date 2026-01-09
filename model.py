import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
MODEL_FILE = os.path.join(BASE_DIR, "cbb_model_v1.pkl")

def train_and_evaluate():
    print("--- 🤖 TRAINING CBB MODEL (HONEST MODE) 🤖 ---")
    
    if not os.path.exists(DATA_FILE):
        print("❌ No processed data found. Run features.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    print(f"   -> Loaded {len(df)} rows.")

    # 2. Define Features (Must match what features.py created)
    # Note: features.py now calculates these directly.
    features = [
        'is_home', 
        'spread', 
        'rest_days', 
        'diff_eFG', 
        'diff_Rebound', 
        'diff_TO', 
        'momentum_gap',
        'roll5_cover_margin'
    ]
    
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

    # 5. Split & Train
    # We shuffle=False to respect time (train on old, test on new) for a quick sanity check
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Random Forest (Standard Config)
    clf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=5, 
        min_samples_leaf=5, 
        random_state=42, 
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # 6. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n   🎯 Validation Accuracy (Holdout): {acc:.1%}")
    # print(classification_report(y_test, preds)) # Optional detail
    
    # 7. Save
    joblib.dump(clf, MODEL_FILE)
    print(f"✅ Final Model Saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_evaluate()