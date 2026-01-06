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
    train_data = df[df['date'] < cutoff_date].dropna()
    df_train = train_data.copy()
    
    df_train['diff_eFG'] = df_train['season_team_eFG'] - df_train['opp_season_team_eFG']
    df_train['diff_Rebound'] = df_train['season_team_ORB'] - df_train['opp_season_team_ORB']
    df_train['diff_TO'] = df_train['season_team_TO'] - df_train['opp_season_team_TO']
    df_train['momentum_gap'] = df_train['roll3_team_eFG'] - df_train['season_team_eFG']
    
    features = ['is_home', 'spread', 'rest_days', 'diff_eFG', 'diff_Rebound', 'diff_TO', 'momentum_gap', 'roll5_cover_margin']
    
    valid_feats = [f for f in features if f in df_train.columns]
    X = df_train[valid_feats]
    y = df_train['ats_win']
    X.columns = X.columns.astype(str)
    
    clf = RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=4, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    return clf, valid_feats

def run_backtest():
    print(f"--- 📉 STARTING BACKTEST (Saving to {OUTPUT_FILE}) ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ CRITICAL ERROR: Training data not found at {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # --- THE FIX IS HERE ---
    # We add 1 day to the max date to ensure the loop includes the final day's games
    end_date = df['date'].max() + timedelta(days=1)
    
    start_date = end_date - timedelta(weeks=WEEKS_BACK)
    
    print(f"   -> Testing Range: {start_date.date()} to {end_date.date()}")
    
    current_date = start_date
    logs = []
    
    while current_date < end_date:
        next_week = current_date + timedelta(days=7)
        # print(f"   -> Window: {current_date.date()} ...") 
        
        model, feats = train_model_at_date(df, current_date)
        
        mask = (df['date'] >= current_date) & (df['date'] < next_week) & (df['is_home'] == 1)
        week_df = df[mask].copy()
        
        if len(week_df) == 0:
            current_date = next_week
            continue
            
        week_df['diff_eFG'] = week_df['season_team_eFG'] - week_df['opp_season_team_eFG']
        week_df['diff_Rebound'] = week_df['season_team_ORB'] - week_df['opp_season_team_ORB']
        week_df['diff_TO'] = week_df['season_team_TO'] - week_df['opp_season_team_TO']
        week_df['momentum_gap'] = week_df['roll3_team_eFG'] - week_df['season_team_eFG']
        
        X_test = week_df[feats].fillna(0)
        X_test.columns = X_test.columns.astype(str)
        
        probs = model.predict_proba(X_test)[:, 1]
        
        week_df['prob_home'] = probs
        week_df['conf'] = week_df['prob_home'].apply(lambda x: max(x, 1-x))
        
        conditions = [week_df['prob_home'] > 0.5, week_df['prob_home'] <= 0.5]
        week_df['picked_team'] = np.select(conditions, [week_df['team'], week_df['opponent']])
        week_df['picked_spread'] = np.select(conditions, [week_df['spread'], -1 * week_df['spread']])
        
        week_df['pick_correct'] = np.where(week_df['prob_home'] > 0.5, week_df['ats_win'] == 1, week_df['ats_win'] == 0)
        
        logs.append(week_df[['date', 'picked_team', 'picked_spread', 'conf', 'pick_correct']])
        current_date = next_week

    if logs:
        full_log = pd.concat(logs)
        action_log = full_log[full_log['conf'] >= 0.53].copy()
        action_log.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ SUCCESS: Saved {len(action_log)} bets to {OUTPUT_FILE}")
    else:
        print("⚠️ WARNING: Backtest ran but generated no bets.")

if __name__ == "__main__":
    run_backtest()