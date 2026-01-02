import pandas as pd
import numpy as np

# --- CONFIG ---
INPUT_FILE = "cbb_training_data_raw.csv"
OUTPUT_FILE = "cbb_training_data_processed.csv"

# STRICTLY SPREAD STATS (No Totals)
STAT_COLS = [
    'team_eFG', 'team_TO', 'team_ORB', 'team_FTR', 'team_3PR',
    'opp_eFG', 'opp_TO', 'opp_ORB', 'opp_FTR', 'opp_3PR',
    'team_score', 'opp_score'
]

def calculate_rolling_stats(df, window, label):
    rolling_feats = df.groupby('team')[STAT_COLS].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    rolling_feats = rolling_feats.add_prefix(f"{label}_")
    
    # Volatility (Standard Deviation)
    vol_feat = df.groupby('team')['team_score'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=3).std()
    )
    rolling_feats[f"{label}_score_volatility"] = vol_feat.fillna(10.0)
    
    return rolling_feats

def calculate_season_to_date(df):
    expanding_feats = df.groupby(['season', 'team'])[STAT_COLS].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    expanding_feats = expanding_feats.add_prefix("season_")
    return expanding_feats

def add_rest_days(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['team', 'date'])
    df['last_game_date'] = df.groupby('team')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['last_game_date']).dt.days
    df['rest_days'] = df['rest_days'].fillna(7).clip(upper=7)
    return df.drop(columns=['last_game_date'])

def main():
    print("--- 🧠 FEATURE ENGINEERING (DATA RESCUE MODE) 🧠 ---")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("❌ Error: cbb_training_data_raw.csv not found."); return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['date', 'team'])
    
    # Clean Team Names (Strip whitespace) to fix Merge issues
    df['team'] = df['team'].str.strip()
    df['opponent'] = df['opponent'].str.strip()

    df = add_rest_days(df)
    
    # 1. Base Stats
    season_stats = calculate_season_to_date(df)
    df = pd.concat([df, season_stats], axis=1)
    
    # 2. Momentum
    momentum_stats = calculate_rolling_stats(df, window=5, label="roll5")
    df = pd.concat([df, momentum_stats], axis=1)
    
    hot_stats = calculate_rolling_stats(df, window=3, label="roll3")
    df = pd.concat([df, hot_stats], axis=1)
    
    # 3. Target (ATS Win)
    df['margin'] = df['team_score'] - df['opp_score']
    if 'spread' in df.columns:
        df['cover_margin'] = df['margin'] + df['spread']
        df['ats_win'] = (df['cover_margin'] > 0).astype(int)
        df['roll5_cover_margin'] = df.groupby('team')['cover_margin'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
    else:
        df['roll5_cover_margin'] = 0.0
    
    # --- DATA RESCUE: FILL NA instead of DROP ---
    # Fills missing early-season rolling stats with 0 (neutral)
    cols_to_fill = [c for c in df.columns if 'roll' in c]
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    
    # 4. Merge Opponent
    # Only drop rows where we have NO season stats (First game of season)
    df_final = df.dropna(subset=['season_team_eFG'])
    
    print("   -> Merging opponent pre-game stats...")
    cols_to_merge = ['date', 'team'] + [c for c in df_final.columns if 'season_' in c or 'roll' in c]
    df_opp = df_final[cols_to_merge].copy().rename(columns={'team': 'opponent'})
    
    for col in df_opp.columns:
        if 'season_' in col or 'roll' in col:
            df_opp = df_opp.rename(columns={col: f"opp_{col}"})
            
    df_model = pd.merge(df_final, df_opp, on=['date', 'opponent'], how='left')
    
    # FINAL CLEAN: If merge failed (opponent not found), drop.
    # But print how many we kept!
    before_drop = len(df_model)
    df_model = df_model.dropna(subset=['opp_season_team_eFG'])
    after_drop = len(df_model)
    
    print(f"   -> Rows before Opponent Check: {before_drop}")
    print(f"   -> Rows after Opponent Check:  {after_drop}")
    print(f"   -> Dropped {before_drop - after_drop} rows due to merge failure.")
    
    df_model.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ SUCCESS: Dataset Rebuilt ({len(df_model)} games).")

if __name__ == "__main__":
    main()