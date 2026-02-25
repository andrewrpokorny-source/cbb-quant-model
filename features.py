import argparse
import os
import sys

import numpy as np
import pandas as pd

from league_config import get_league_artifact_paths, normalize_league

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

    # Recompute rest days from schedule history so training inputs are consistent
    # across leagues, even if raw rows do not include a rest_days field.
    df['prev_game_date'] = df.groupby('team')['date'].shift(1)
    rest = (pd.to_datetime(df['date']) - pd.to_datetime(df['prev_game_date'])).dt.days
    df['rest_days'] = rest.fillna(7).clip(lower=0, upper=7)
    df = df.drop(columns=['prev_game_date'])

    stats_cols = ['eFG', 'TS', 'off_rating', 'poss', 'orb', 'to', 'team_score']

    for col in stats_cols:
        df[f'season_team_{col}'] = df.groupby('team')[col].expanding().mean().reset_index(level=0, drop=True)
        df[f'roll3_team_{col}'] = df.groupby('team')[col].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    for col in stats_cols:
        df[f'prev_season_{col}'] = df.groupby('team')[f'season_team_{col}'].shift(1)
        df[f'prev_roll3_{col}'] = df.groupby('team')[f'roll3_team_{col}'].shift(1)

    # --- Cover Margin ---
    df['cover_margin'] = df['team_score'] + df['spread'] - df['opp_score']
    roll_vals = df.groupby('team')['cover_margin'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['roll5_cover_margin'] = roll_vals
    df['roll5_cover_margin'] = df.groupby('team')['roll5_cover_margin'].shift(1)

    # --- NEW FEATURES (V2 Improvements) ---
    print("   -> Adding V2 features (games_played, volatility, margin, blowout, opp_quality)...")

    # 1. Games Played (sample size indicator)
    df['games_played'] = df.groupby('team').cumcount()
    df['prev_games_played'] = df.groupby('team')['games_played'].shift(1).fillna(0)

    # 2. Score Volatility (consistency indicator)
    df['roll5_score_std'] = df.groupby('team')['team_score'].rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    df['prev_volatility'] = df.groupby('team')['roll5_score_std'].shift(1).fillna(10)

    # 3. Point Margin (recent performance)
    df['margin'] = df['team_score'] - df['opp_score']
    df['roll5_margin'] = df.groupby('team')['margin'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['prev_roll5_margin'] = df.groupby('team')['roll5_margin'].shift(1).fillna(0)

    # 4. Blowout Rate (dominance indicator)
    df['blowout_win'] = (df['margin'] > 15).astype(int)
    df['roll5_blowout_rate'] = df.groupby('team')['blowout_win'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df['prev_blowout_rate'] = df.groupby('team')['roll5_blowout_rate'].shift(1).fillna(0)

    # 5. Win tracking for opponent quality
    df['game_win'] = (df['team_score'] > df['opp_score']).astype(int)
    df['season_wins'] = df.groupby('team')['game_win'].cumsum()
    df['win_pct'] = df['season_wins'] / (df['games_played'] + 1).clip(lower=1)
    df['prev_win_pct'] = df.groupby('team')['win_pct'].shift(1).fillna(0.5)

    return df

def merge_opponent_stats(df):
    print("   -> Merging opponent entering stats...")

    req_cols = ['date', 'team', 'prev_season_eFG', 'prev_season_orb', 'prev_season_to', 'prev_season_off_rating', 'prev_win_pct']
    opp_lookup = df[req_cols].copy()

    rename_map = {
        'team': 'opponent_name',
        'prev_season_eFG': 'opp_season_team_eFG',
        'prev_season_orb': 'opp_season_team_ORB',
        'prev_season_to': 'opp_season_team_TO',
        'prev_season_off_rating': 'opp_season_off_rating',
        'prev_win_pct': 'opp_win_pct'
    }
    opp_lookup = opp_lookup.rename(columns=rename_map)

    df_merged = pd.merge(df, opp_lookup, left_on=['date', 'opponent'], right_on=['date', 'opponent_name'], how='left', suffixes=('', '_dupe'))

    df_merged['diff_eFG'] = df_merged['prev_season_eFG'] - df_merged['opp_season_team_eFG']
    df_merged['diff_Rebound'] = df_merged['prev_season_orb'] - df_merged['opp_season_team_ORB']
    df_merged['diff_TO'] = df_merged['prev_season_to'] - df_merged['opp_season_team_TO']
    df_merged['momentum_gap'] = df_merged['prev_roll3_eFG'] - df_merged['prev_season_eFG']

    # Fill missing opponent win pct with 0.5 (neutral)
    df_merged['opp_win_pct'] = df_merged['opp_win_pct'].fillna(0.5)

    if 'opponent_name' in df_merged.columns:
        df_merged = df_merged.drop(columns=['opponent_name'])

    return df_merged

def main(league="mens"):
    league = normalize_league(league)
    data_file = get_league_artifact_paths(BASE_DIR, league)["data_file"]

    print(f"--- 🧠 FEATURE ENGINEERING (HONEST MODE: FIXED, {league}) 🧠 ---")
    if not os.path.exists(data_file):
        print("❌ No data file found."); return

    # Suppress Mixed Type Warning
    df = pd.read_csv(data_file, low_memory=False)
    
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
    df_final.to_csv(data_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate engineered features for CBB model training.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League to process: mens or womens (aliases supported).",
    )
    args = parser.parse_args()
    main(args.league)
