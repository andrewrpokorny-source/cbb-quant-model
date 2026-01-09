import pandas as pd
import numpy as np
import os
import sys

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")

def clean_stale_data(df):
    """Removes old calculated columns to prevent merge conflicts."""
    print("   -> 🧹 Cleaning stale columns...")
    # List of keywords that indicate a calculated column
    keywords = ['season_', 'roll', 'prev_', 'opp_', 'diff_', 'eFG', 'TS', 'off_rating', 'poss', 'ats_win']
    
    keep_cols = ['date', 'team', 'opponent', 'location', 'team_score', 'opp_score', 'spread', 'is_home']
    
    # Identify core columns (keep these) and calculated columns (drop these)
    current_cols = df.columns.tolist()
    drop_list = []
    
    for col in current_cols:
        if col in keep_cols: continue
        # If it looks like a calculated stat, drop it so we can recalculate fresh
        if any(k in col for k in keywords) or col in ['fga', 'to', 'fta', 'orb', 'fgm', '3pm']:
            drop_list.append(col)
            
    if drop_list:
        df = df.drop(columns=drop_list)
    
    return df

def calculate_advanced_stats(df):
    print("   -> Calculating Possessions & Efficiency...")
    # Re-estimate basic stats if they were dropped or missing
    # (Since we cleaned them, we must regenerate or ensure they exist)
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
    
    # 1. SORT & RESET INDEX
    df = df.sort_values(['team', 'date'])
    df = df.reset_index(drop=True) 
    
    stats_cols = ['eFG', 'TS', 'off_rating', 'poss', 'orb', 'to', 'team_score']
    
    # 2. CALCULATE ROLLING AVERAGES
    for col in stats_cols:
        # Expanding Mean (Season to Date)
        # We use simple .groupby().transform() or explicit mapping to avoid MultiIndex issues
        df[f'season_team_{col}'] = df.groupby('team')[col].expanding().mean().reset_index(level=0, drop=True)
        # Rolling 3 (Momentum)
        df[f'roll3_team_{col}'] = df.groupby('team')[col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        
    # 3. CREATE "ENTERING" STATS (The Anti-Cheat Fix)
    for col in stats_cols:
        df[f'prev_season_{col}'] = df.groupby('team')[f'season_team_{col}'].shift(1)
        df[f'prev_roll3_{col}'] = df.groupby('team')[f'roll3_team_{col}'].shift(1)
        
    # Cover Margin Logic
    df['cover_margin'] = df['team_score'] + df['spread'] - df['opp_score']
    
    # Rolling cover margin
    roll_margin = df.groupby('team')['cover_margin'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['roll5_cover_margin'] = roll_margin
    # Shift to ensure we only know the PAST cover margin
    df['roll5_cover_margin'] = df.groupby('team')['roll5_cover_margin'].shift(1)
    
    return df

def merge_opponent_stats(df):
    print("   -> Merging opponent entering stats...")
    
    # Explicitly select the columns we need
    # Note: 'orb' and 'to' are lowercase in our generation loop above
    req_cols = ['date', 'team', 'prev_season_eFG', 'prev_season_orb', 'prev_season_to', 'prev_season_off_rating']
    
    # Verify columns exist before grabbing them
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"❌ CRITICAL: Missing columns before merge: {missing}")
        print("   (This usually means calculate_rolling_stats failed silently)")
        sys.exit(1)

    opp_lookup = df[req_cols].copy()
    
    # Safer Dictionary Rename
    rename_map = {
        'team': 'opponent_name',
        'prev_season_eFG': 'opp_season_team_eFG',
        'prev_season_orb': 'opp_season_team_ORB',
        'prev_season_to': 'opp_season_team_TO',
        'prev_season_off_rating': 'opp_season_off_rating'
    }
    opp_lookup = opp_lookup.rename(columns=rename_map)
    
    # Merge
    # We use suffixes=('', '_dupe') just in case, but since we cleaned data, it should be fine.
    df_merged = pd.merge(df, opp_lookup, left_on=['date', 'opponent'], right_on=['date', 'opponent_name'], how='left', suffixes=('', '_dupe'))
    
    # Check if column exists
    if 'opp_season_team_eFG' not in df_merged.columns:
        print("❌ Merge Failed: 'opp_season_team_eFG' missing.")
        print("   Columns found:", df_merged.columns.tolist())
        sys.exit(1)

    # --- CALCULATE HONEST DIFFERENTIALS ---
    df_merged['diff_eFG'] = df_merged['prev_season_eFG'] - df_merged['opp_season_team_eFG']
    df_merged['diff_Rebound'] = df_merged['prev_season_orb'] - df_merged['opp_season_team_ORB']
    df_merged['diff_TO'] = df_merged['prev_season_to'] - df_merged['opp_season_team_TO']
    df_merged['momentum_gap'] = df_merged['prev_roll3_eFG'] - df_merged['prev_season_eFG']
    
    # Drop rows where we don't have stats (start of season)
    df_merged = df_merged.dropna(subset=['diff_eFG'])
    
    # Clean up the extra opponent_name column
    if 'opponent_name' in df_merged.columns:
        df_merged = df_merged.drop(columns=['opponent_name'])
    
    return df_merged

def main():
    print("--- 🧠 FEATURE ENGINEERING (HONEST MODE: V2 ROBUST) 🧠 ---")
    if not os.path.exists(DATA_FILE):
        print("❌ No data file found."); return

    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # CLEAN SLATE
    df = clean_stale_data(df)
    
    # Ensure numerics
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