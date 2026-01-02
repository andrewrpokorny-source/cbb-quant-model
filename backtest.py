import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from datetime import timedelta
import os

# --- CONFIG ---
DATA_FILE = "cbb_training_data_processed.csv"
OUTPUT_FILE = "performance_log.csv"
WEEKS_BACK = 4

def train_model_at_date(df, cutoff_date):
    """Trains a model strictly on data BEFORE the cutoff_date."""
    # Split data
    train_data = df[df['date'] < cutoff_date].dropna()
    
    # Feature Engineering (Same as production)
    df_train = train_data.copy()
    
    # Calculate Original Features
    df_train['diff_eFG'] = df_train['season_team_eFG'] - df_train['opp_season_team_eFG']
    df_train['diff_Rebound'] = df_train['season_team_ORB'] - df_train['opp_season_team_ORB']
    df_train['diff_TO'] = df_train['season_team_TO'] - df_train['opp_season_team_TO']
    df_train['momentum_gap'] = df_train['roll3_team_eFG'] - df_train['season_team_eFG']
    
    # Core 8 Features
    features = [
        'is_home', 'spread', 'rest_days', 
        'diff_eFG', 'diff_Rebound', 'diff_TO', 
        'momentum_gap', 'roll5_cover_margin'
    ]
    
    # Validation check
    valid_feats = [f for f in features if f in df_train.columns]
    X = df_train[valid_feats]
    y = df_train['ats_win']
    
    # Force String Cols
    X.columns = X.columns.astype(str)
    
    # Train RF
    clf = RandomForestClassifier(
        n_estimators=500, max_depth=7, min_samples_leaf=4, 
        random_state=42, n_jobs=-1
    )
    clf.fit(X, y)
    
    return clf, valid_feats

def run_backtest():
    print(f"--- 📉 STARTING {WEEKS_BACK}-WEEK HONEST BACKTEST ---")
    
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    except:
        print("❌ Data missing. Run features.py first."); return

    # Define Time Range
    end_date = df['date'].max()
    start_date = end_date - timedelta(weeks=WEEKS_BACK)
    
    print(f"   -> Testing Range: {start_date.date()} to {end_date.date()}")
    
    current_date = start_date
    logs = []
    
    while current_date < end_date:
        next_week = current_date + timedelta(days=7)
        print(f"   -> Window: {current_date.date()} ...")
        
        # 1. Train Model
        model, feats = train_model_at_date(df, current_date)
        
        # 2. Predict 'Next Week'
        # CRITICAL FIX: Filter to is_home=1 to avoid double counting and confusion
        mask = (df['date'] >= current_date) & (df['date'] < next_week) & (df['is_home'] == 1)
        week_df = df[mask].copy()
        
        if len(week_df) == 0:
            current_date = next_week
            continue
            
        # Feature Prep
        week_df['diff_eFG'] = week_df['season_team_eFG'] - week_df['opp_season_team_eFG']
        week_df['diff_Rebound'] = week_df['season_team_ORB'] - week_df['opp_season_team_ORB']
        week_df['diff_TO'] = week_df['season_team_TO'] - week_df['opp_season_team_TO']
        week_df['momentum_gap'] = week_df['roll3_team_eFG'] - week_df['season_team_eFG']
        
        X_test = week_df[feats].fillna(0)
        X_test.columns = X_test.columns.astype(str)
        
        # Predict
        probs = model.predict_proba(X_test)[:, 1]
        
        # --- THE LOGIC FIX ---
        week_df['prob_home'] = probs
        week_df['conf'] = week_df['prob_home'].apply(lambda x: max(x, 1-x))
        
        # If Prob(Home) > 0.5: Pick Home.
        # If Prob(Home) < 0.5: Pick Away.
        
        conditions = [week_df['prob_home'] > 0.5, week_df['prob_home'] <= 0.5]
        
        # Name of team we are betting ON
        week_df['picked_team'] = np.select(conditions, [week_df['team'], week_df['opponent']])
        
        # Spread we are getting (Home spread, or inverted Away spread)
        week_df['picked_spread'] = np.select(conditions, [week_df['spread'], -1 * week_df['spread']])
        
        # Did we win?
        # If picking Home: Win if ats_win == 1
        # If picking Away: Win if ats_win == 0
        week_df['pick_correct'] = np.where(
            week_df['prob_home'] > 0.5, 
            week_df['ats_win'] == 1, 
            week_df['ats_win'] == 0
        )
        
        # Log clean columns
        logs.append(week_df[['date', 'picked_team', 'picked_spread', 'conf', 'pick_correct']])
        
        current_date = next_week

    full_log = pd.concat(logs)
    
    # Filter for Actionable bets
    action_log = full_log[full_log['conf'] >= 0.53].copy()
    
    print(f"\n✅ Backtest Complete.")
    print(f"   -> Actionable Bets: {len(action_log)}")
    
    acc = action_log['pick_correct'].mean()
    print(f"   -> 🏆 Realized Accuracy: {acc:.1%}")
    
    action_log.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    run_backtest()