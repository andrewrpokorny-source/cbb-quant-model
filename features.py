import pandas as pd
import numpy as np
import os
import sys

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")

def clean_stale_data(df):
    print("   -> 🧹 Cleaning stale columns...")
    keywords = ['season_', 'roll', 'prev_', 'opp_', 'diff_', 'eFG', 'TS', 'off_rating', 'poss', 'ats_win']
    keep_cols = ['date', 'team', 'opponent', 'location', 'team_score', 'opp_score', 'spread', 'is_home']
    
    current_cols = df.columns.tolist()
    drop_list = []
    for col in current_cols:
        if col in keep_cols: continue
        if any(k in col for k in keywords) or col in ['fga', 'to', 'fta', 'orb', 'fgm', '3pm']:
            drop_list.append(col)
            
    if drop_list:
        df = df.drop(columns=drop_list)
    return df

def calculate_advanced_stats(df):
    print("   -> Calculating Possessions & Efficiency...")
    if 'fga' not in df.columns:
        df['fga'] = df['team_score'] / 2
        df['to'] = 12
        df['fta'] = df['team_score'] / 4
        df['orb'] = 8
        df['fgm'] = df['team_score'] / 2.2
        df['3pm'] = 6
        
    df['poss'] = 0.96 * (df['fga'] + df['to'] + 0.44 * df['fta'] - df['orb'])
    df['off_rating'] = 100 * (df['team_score'] / df['poss'])
    df['eFG'] = (df['fgm'] + 0.5 * df['3pm']) / df['fga']
    df['TS'] = df['team_score'] / (2 * (df['fga'] + 0.44 * df['fta']))
    return df

def calculate_rolling_stats(df):
    print("   -> Generating Rolling Averages (Honest Lag)...")
    df = df.sort_values(['team', 'date']).reset_index(drop=True)
    
    stats_cols = ['eFG', 'TS', 'off_rating', 'poss', 'orb', 'to', 'team_score']
    
    for col in stats_cols:
        df[f'season_team_{col}'] = df.groupby('team')[col].expanding().mean().reset_index(level=0, drop=True)
        df[f'roll3_team_{col}'] = df.groupby('team')[col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        
    for col in stats_cols:
        df[f'prev_season_{col}'] = df.groupby('team')[f'season_team_{col}'].shift(1)
        df[f'prev_roll3_{col}'] = df.groupby('team')[f'roll3_team_{col}'].shift(1)
        
    # --- FIX IS HERE ---
    df['cover_margin'] = df['team_score'] + df['spread'] - df['opp_score']
    
    # 1. Calculate the rolling stats
    roll_vals = df.groupby('team')['cover_margin'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # 2. Assign to DF (So the column exists!)
    df['roll5_cover_margin'] = roll_vals
    
    # 3. NOW shift it (Safe because the column exists)
    df['roll5_cover_margin'] = df.groupby('team')['roll5_cover_margin'].shift(1)
    
    return df

def merge_opponent_stats(df):
    print("   -> Merging opponent entering stats...")
    
    req_cols = ['date', 'team', 'prev_season_eFG', 'prev_season_orb', 'prev_season_to', 'prev_season_off_rating']
    opp_lookup = df[req_cols].copy()
    
    rename_map = {
        'team': 'opponent_name',
        'prev_season_eFG': 'opp_season_team_eFG',
        'prev_season_orb': 'opp_season_team_ORB',
        'prev_season_to': 'opp_season_team_TO',
        'prev_season_off_rating': 'opp_season_off_rating'
    }
    opp_lookup = opp_lookup.rename(columns=rename_map)
    
    df_merged = pd.merge(df, opp_lookup, left_on=['date', 'opponent'], right_on=['date', 'opponent_name'], how='left', suffixes=('', '_dupe'))
    
    df_merged['diff_eFG'] = df_merged['prev_season_eFG'] - df_merged['opp_season_team_eFG']
    df_merged['diff_Rebound'] = df_merged['prev_season_orb'] - df_merged['opp_season_team_ORB']
    df_merged['diff_TO'] = df_merged['prev_season_to'] - df_merged['opp_season_team_TO']
    df_merged['momentum_gap'] = df_merged['prev_roll3_eFG'] - df_merged['prev_season_eFG']
    
    if 'opponent_name' in df_merged.columns:
        df_merged = df_merged.drop(columns=['opponent_name'])
        
    return df_merged

def main():
    print("--- 🧠 FEATURE ENGINEERING (HONEST MODE: FIXED) 🧠 ---")
    if not os.path.exists(DATA_FILE):
        print("❌ No data file found."); return

    # Suppress Mixed Type Warning
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # Normalize Dates
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    df = clean_stale_data(df)
    
    cols = ['team_score', 'opp_score', 'spread']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df = calculate_advanced_stats(df)
    df = calculate_rolling_stats(df)
    df['ats_win'] = (df['team_score'] + df['spread'] > df['opp_score']).astype(int)
    
    df_final = merge_opponent_stats(df)
    
    print(f"✅ Saving processed data ({len(df_final)} rows)...")
    df_final.to_csv(DATA_FILE, index=False)

if __name__ == "__main__":
    main()