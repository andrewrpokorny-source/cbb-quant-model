import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib

# --- CONFIG ---
DATA_FILE = "cbb_training_data_processed.csv"
MODEL_FILE = "cbb_model_v1.pkl"

def create_original_features(df):
    """
    Restores the EXACT logic from the 53.7% run.
    """
    # 1. Simple Efficiency Differential
    df['diff_eFG'] = df['season_team_eFG'] - df['opp_season_team_eFG']
    df['diff_Rebound'] = df['season_team_ORB'] - df['opp_season_team_ORB']
    df['diff_TO'] = df['season_team_TO'] - df['opp_season_team_TO']
    
    # 2. Momentum Gap
    df['momentum_gap'] = df['roll3_team_eFG'] - df['season_team_eFG']
    
    return df

def train_and_evaluate():
    print("--- 🤖 TRAINING CBB MODEL (SURGICAL CLEANING) 🤖 ---")
    
    try:
        df = pd.read_csv(DATA_FILE)
        raw_len = len(df)
        print(f"   -> Loaded {raw_len} raw rows.")
    except:
        print("❌ Error: Data not found."); return

    # 1. Create Features on the FULL set first
    df = create_original_features(df)

    # 2. Define The Core Features
    feature_cols = [
        'is_home', 
        'spread', 
        'rest_days', 
        'diff_eFG', 
        'diff_Rebound', 
        'diff_TO', 
        'momentum_gap',
        'roll5_cover_margin'
    ]
    
    # 3. SURGICAL SELECTION
    # We only keep the columns we need + the target
    keep_cols = feature_cols + ['ats_win']
    
    # Filter to ensure columns exist
    final_cols = [c for c in keep_cols if c in df.columns]
    
    df_clean = df[final_cols].copy()
    
    # 4. SURGICAL DROP
    # Now we drop rows ONLY if one of OUR features is missing
    df_clean = df_clean.dropna()
    clean_len = len(df_clean)
    
    print(f"   -> Used {clean_len} rows for training (Recovered {clean_len - 5194} games!)")

    X = df_clean[[c for c in feature_cols if c in df_clean.columns]]
    y = df_clean['ats_win']
    
    # Safety: Force strings for Sklearn 1.8.0
    X.columns = X.columns.astype(str)

    # 5. Hyper-Tuned Random Forest
    model = RandomForestClassifier(
        n_estimators=500, 
        max_depth=7, 
        min_samples_leaf=4, 
        random_state=42,
        n_jobs=-1
    )
    
    # Validation
    tscv = TimeSeriesSplit(n_splits=5)
    accs = []
    
    print("\n   -> 🏃‍♂️ Verifying Accuracy...")
    for train_index, test_index in tscv.split(X):
        model.fit(X.iloc[train_index], y.iloc[train_index])
        preds = model.predict(X.iloc[test_index])
        acc = accuracy_score(y.iloc[test_index], preds)
        accs.append(acc)
        # print(f"      Fold {len(accs)}: {acc:.1%}")
        
    avg_acc = np.mean(accs)
    print(f"   🎯 Validation Accuracy: {avg_acc:.1%}")

    # Final Save
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("✅ Final Model Saved.")

if __name__ == "__main__":
    train_and_evaluate()