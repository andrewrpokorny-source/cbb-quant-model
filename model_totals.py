import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_FILE = "cbb_training_data_processed.csv"
MODEL_FILE = "cbb_totals_model.pkl"

def create_totals_features(df):
    """Features specific to Over/Under Prediction."""
    # 1. Projected Pace
    if 'season_possessions' in df.columns and 'opp_season_possessions' in df.columns:
        df['combined_pace'] = df['season_possessions'] + df['opp_season_possessions']
    else:
        df['combined_pace'] = 140.0 

    # 2. Pace Momentum
    if 'roll3_possessions' in df.columns:
        df['pace_momentum'] = (df['roll3_possessions'] + df['opp_roll3_possessions']) - df['combined_pace']
    else:
        df['pace_momentum'] = 0.0

    # 3. Efficiency Matchup
    matchup_a = df['season_team_eFG'] - df['opp_season_opp_eFG'] 
    matchup_b = df['opp_season_team_eFG'] - df['season_opp_eFG']
    df['scoring_environment'] = matchup_a + matchup_b
    
    # 4. Foul Rate
    df['combined_ftr'] = df['season_team_FTR'] + df['opp_season_team_FTR']
    
    # 5. Line Value
    if 'total_line' in df.columns:
        safe_line = df['total_line'].replace(0, 140)
        df['pace_to_line_ratio'] = df['combined_pace'] / safe_line
    else:
        df['pace_to_line_ratio'] = 1.0

    # 6. Volatility Differential
    if 'diff_volatility' not in df.columns:
        if 'roll5_score_volatility' in df.columns and 'opp_roll5_score_volatility' in df.columns:
            df['diff_volatility'] = df['roll5_score_volatility'] - df['opp_roll5_score_volatility']
        else:
            df['diff_volatility'] = 0.0

    if 'roll5_over_margin' not in df.columns:
        df['roll5_over_margin'] = 0.0
        
    return df

def train_totals():
    print("--- 🤖 TRAINING CBB TOTALS MODEL (OVER/UNDER) 🤖 ---")
    
    try:
        df = pd.read_csv(DATA_FILE)
        if 'total_over' not in df.columns:
            print("   ⚠️ 'total_over' column missing. Re-run features.py.")
            return

        # Filter for valid data
        df = df.dropna(subset=['total_line', 'total_over'])
        df = df[df['total_line'] > 100] 
    except FileNotFoundError:
        print("❌ Data file not found."); return

    # --- SAFETY CHECK: DATA VOLUME ---
    # We need at least 500 games to train a complex XGBoost model.
    # If we have less, we fallback to a simple Random Forest to prevent crashes.
    n_samples = len(df)
    print(f"   -> Found {n_samples} valid games with Totals.")
    
    if n_samples < 100:
        print("   ❌ ERROR: Not enough data points (< 100). Check your scraping (main.py).")
        print("   -> Aborting Totals training to prevent bad model.")
        return

    df = create_totals_features(df)
    
    feature_cols = [
        'combined_pace', 'pace_momentum', 'scoring_environment', 
        'combined_ftr', 'pace_to_line_ratio',
        'roll5_over_margin', 'diff_volatility'
    ]
    
    X = df[feature_cols]
    y = df['total_over']

    # --- SAFETY CHECK: CLASS BALANCE ---
    # XGBoost crashes if y contains only 0s or only 1s.
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        print(f"   ❌ ERROR: Target variable has only one class! {class_counts}")
        return
        
    print(f"   -> Training Features: {len(feature_cols)}")

    # LOGIC: If small data, use RF only. If big data, use Ensemble.
    if n_samples < 1000:
        print("   ⚠️ Small Dataset detected. Using Random Forest ONLY (Safe Mode).")
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    else:
        print("   🚀 Big Data detected. Using Full Ensemble (RF + XGB).")
        rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
        xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb_clf)], voting='soft')

    # Validation
    tscv = TimeSeriesSplit(n_splits=5)
    accs = []
    
    print("   -> 🏃‍♂️ Validating...")
    for train, test in tscv.split(X):
        # Double check split validity
        y_train = y.iloc[train]
        if len(y_train.unique()) < 2: continue # Skip bad folds
        
        model.fit(X.iloc[train], y_train)
        preds = model.predict(X.iloc[test])
        acc = accuracy_score(y.iloc[test], preds)
        accs.append(acc)
        print(f"      Fold Acc: {acc:.1%}")
        
    if accs:
        print(f"   🎯 Totals Accuracy: {np.mean(accs):.1%}")
    
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("✅ Totals Model Saved.")

if __name__ == "__main__":
    train_totals()