import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
import os

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "performance_log.csv")
WEEKS_BACK = 4

def train_model_at_date(df, cutoff_date):
    # 1. Filter for PAST games only
    # We create a copy to avoid SettingWithCopy warnings
    past_games = df[df['date'] < cutoff_date].copy()
    
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
    
    # Check 1: Do we have the columns?
    valid_feats = [f for f in features if f in past_games.columns]
    if not valid_feats:
        return None, None

    # Check 2: Drop NaNs only for the features we need
    # This keeps us from dropping rows just because unrelated columns are empty
    train_data = past_games.dropna(subset=valid_feats + ['ats_win'])
    
    # Check 3: Do we have enough data to actually learn?
    # Random Forest needs at least a decent sample size.
    if len(train_data) < 50:
        return None, None
    
    X = train_data[valid_feats]
    y = train_data['ats_win']
    X.columns = X.columns.astype(str)
    
    # 3. Train Model
    clf = RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=4, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    return clf, valid_feats

def run_backtest():
    print(f"--- 📉 STARTING BACKTEST (Honest Mode) ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ CRITICAL ERROR: Training data not found at {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # --- CALCULATE REST DAYS (Dynamic) ---
    df['last_game'] = df.groupby('team')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['last_game']).dt.days.fillna(7)
    df['rest_days'] = df['rest_days'].clip(upper=7) 

    # End Date: Max date in file + 1 day to cover everything
    end_date = df['date'].max() + timedelta(days=1)
    start_date = end_date - timedelta(weeks=WEEKS_BACK)
    
    print(f"   -> Testing Range: {start_date.date()} to {end_date.date()}")
    
    current_date = start_date
    logs = []
    
    while current_date < end_date:
        next_week = current_date + timedelta(days=7)
        
        # Train on EVERYTHING before current_date
        model, feats = train_model_at_date(df, current_date)
        
        # SAFETY: If not enough data, skip this week
        if model is None:
            print(f"      ⚠️  Not enough history to train for week of {current_date.date()}. Skipping.")
            current_date = next_week
            continue
        
        # Test on THIS WEEK (current_date to next_week)
        mask = (df['date'] >= current_date) & (df['date'] < next_week) & (df['is_home'] == 1)
        week_df = df[mask].copy()
        
        if len(week_df) > 0:
            # Drop rows in test set if they miss features (can't predict on partial data)
            week_df = week_df.dropna(subset=feats)
            
            if len(week_df) > 0:
                X_test = week_df[feats]
                X_test.columns = X_test.columns.astype(str)
                
                probs = model.predict_proba(X_test)[:, 1]
                
                week_df['prob_home'] = probs
                week_df['conf'] = week_df['prob_home'].apply(lambda x: max(x, 1-x))
                
                # Logic: If prob_home > 0.5, Pick Home. Else Pick Away.
                conditions = [week_df['prob_home'] > 0.5, week_df['prob_home'] <= 0.5]
                week_df['picked_team'] = np.select(conditions, [week_df['team'], week_df['opponent']])
                week_df['picked_spread'] = np.select(conditions, [week_df['spread'], -1 * week_df['spread']])
                
                # Grade: Did the pick win?
                week_df['pick_correct'] = np.where(week_df['prob_home'] > 0.5, week_df['ats_win'] == 1, week_df['ats_win'] == 0)
                
                logs.append(week_df[['date', 'picked_team', 'picked_spread', 'conf', 'pick_correct']])
        
        current_date = next_week

    if logs:
        full_log = pd.concat(logs)
        # Filter for "Actionable" bets (e.g. > 53% Confidence)
        action_log = full_log[full_log['conf'] >= 0.53].copy()
        action_log.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ SUCCESS: Saved {len(action_log)} bets to {OUTPUT_FILE}")
    else:
        print("⚠️ WARNING: Backtest ran but generated no bets.")

if __name__ == "__main__":
    run_backtest()