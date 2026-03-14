import argparse
import os
import sys

import numpy as np
import pandas as pd

from league_config import get_league_artifact_paths, normalize_league
from odds_archive import load_latest_market_spreads
from hasla import add_hasla_features, ensure_hasla_feature_columns, load_snapshot_file as load_hasla_snapshot_file, load_team_map as load_hasla_team_map
from torvik import add_torvik_features, ensure_torvik_feature_columns, load_snapshot_file, load_team_map
from venue import build_team_home_locations, compute_distance_advantage_bulk, load_geocode_cache, save_geocode_cache
from womens_net import (
    add_womens_net_features,
    ensure_womens_net_feature_columns,
    load_snapshot_file as load_womens_net_snapshot_file,
    load_team_map as load_womens_net_team_map,
)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_COLUMNS = [
    "date",
    "season",
    "total_line",
    "team",
    "opponent",
    "location",
    "team_score",
    "opp_score",
    "spread",
    "is_home",
    "is_neutral",
    "venue_city",
    "venue_state",
]
RAW_RATE_COLUMNS = [
    "possessions",
    "team_eFG",
    "team_TO",
    "team_ORB",
    "team_FTR",
    "team_3PR",
    "opp_eFG",
    "opp_TO",
    "opp_ORB",
    "opp_FTR",
    "opp_3PR",
]
STAT_SOURCE_COLUMNS = {
    "eFG": "team_eFG",
    "to": "team_TO",
    "orb": "team_ORB",
    "poss": "possessions",
}

def clean_stale_data(df):
    print("   -> Cleaning stale columns...")
    keep_cols = set(BASE_COLUMNS + RAW_RATE_COLUMNS)
    current_cols = df.columns.tolist()
    drop_list = [col for col in current_cols if col not in keep_cols]

    if drop_list:
        df = df.drop(columns=drop_list)
    return df

def calculate_advanced_stats(df):
    print("   -> Calculating supported rate stats from upstream inputs...")
    available_stats = []
    missing_sources = []

    for stat_col, source_col in STAT_SOURCE_COLUMNS.items():
        if source_col not in df.columns:
            missing_sources.append(source_col)
            continue
        df[stat_col] = pd.to_numeric(df[source_col], errors='coerce')
        available_stats.append(stat_col)

    if 'poss' in available_stats:
        poss = df['poss'].where(df['poss'] > 0)
        df['off_rating'] = 100 * (pd.to_numeric(df['team_score'], errors='coerce') / poss)
        available_stats.append('off_rating')

    if missing_sources:
        print(
            "      Source stats unavailable; skipping unsupported derived metrics:",
            ", ".join(missing_sources),
        )

    return df, available_stats

def calculate_rolling_stats(df, stat_cols):
    print("   -> Generating Rolling Averages (Honest Lag)...")
    df = df.sort_values(['team', 'date']).reset_index(drop=True)

    # Recompute rest days from schedule history so training inputs are consistent
    # across leagues, even if raw rows do not include a rest_days field.
    df['prev_game_date'] = df.groupby('team')['date'].shift(1)
    rest = (pd.to_datetime(df['date']) - pd.to_datetime(df['prev_game_date'])).dt.days
    df['rest_days'] = rest.fillna(7).clip(lower=0, upper=7)
    df = df.drop(columns=['prev_game_date'])

    stats_cols = list(dict.fromkeys(stat_cols + ['team_score']))

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

    req_cols = ['date', 'team', 'prev_win_pct']
    rename_map = {
        'team': 'opponent_name',
        'prev_win_pct': 'opp_win_pct'
    }
    optional_prev_cols = {
        'prev_season_eFG': 'opp_season_team_eFG',
        'prev_season_orb': 'opp_season_team_ORB',
        'prev_season_to': 'opp_season_team_TO',
        'prev_season_off_rating': 'opp_season_off_rating',
    }
    for src_col, dest_col in optional_prev_cols.items():
        if src_col in df.columns:
            req_cols.append(src_col)
            rename_map[src_col] = dest_col

    opp_lookup = df[req_cols].copy()
    opp_lookup = opp_lookup.rename(columns=rename_map)

    df_merged = pd.merge(df, opp_lookup, left_on=['date', 'opponent'], right_on=['date', 'opponent_name'], how='left', suffixes=('', '_dupe'))

    if {'prev_season_eFG', 'opp_season_team_eFG'}.issubset(df_merged.columns):
        df_merged['diff_eFG'] = (
            df_merged['prev_season_eFG'] - df_merged['opp_season_team_eFG']
        ).fillna(0.0)
    else:
        df_merged['diff_eFG'] = 0.0

    if {'prev_season_orb', 'opp_season_team_ORB'}.issubset(df_merged.columns):
        df_merged['diff_Rebound'] = (
            df_merged['prev_season_orb'] - df_merged['opp_season_team_ORB']
        ).fillna(0.0)
    else:
        df_merged['diff_Rebound'] = 0.0

    if {'prev_season_to', 'opp_season_team_TO'}.issubset(df_merged.columns):
        df_merged['diff_TO'] = (
            df_merged['prev_season_to'] - df_merged['opp_season_team_TO']
        ).fillna(0.0)
    else:
        df_merged['diff_TO'] = 0.0

    if {'prev_roll3_eFG', 'prev_season_eFG'}.issubset(df_merged.columns):
        df_merged['momentum_gap'] = (
            df_merged['prev_roll3_eFG'] - df_merged['prev_season_eFG']
        ).fillna(0.0)
    else:
        df_merged['momentum_gap'] = 0.0

    # Fill missing opponent win pct with 0.5 (neutral)
    df_merged['opp_win_pct'] = df_merged['opp_win_pct'].fillna(0.5)

    if 'opponent_name' in df_merged.columns:
        df_merged = df_merged.drop(columns=['opponent_name'])

    return df_merged


def merge_torvik_priors(df, league, paths):
    df = ensure_torvik_feature_columns(df)
    if league != "mens":
        return df

    snapshot_file = paths.get("torvik_snapshot_file")
    map_file = paths.get("torvik_map_file")
    if not snapshot_file or not map_file:
        return df

    snapshots_df = load_snapshot_file(snapshot_file)
    team_map_df = load_team_map(
        map_file,
        pd.concat([df["team"], df["opponent"]], ignore_index=True).dropna().unique(),
    )
    if snapshots_df.empty or team_map_df.empty:
        return df

    print("   -> Merging Bart Torvik lagged priors...")
    return add_torvik_features(df, snapshots_df, team_map_df)


def merge_hasla_priors(df, league, paths):
    df = ensure_hasla_feature_columns(df)
    if league != "mens":
        return df

    snapshot_file = paths.get("hasla_snapshot_file")
    map_file = paths.get("hasla_map_file")
    if not snapshot_file or not map_file:
        return df

    snapshots_df = load_hasla_snapshot_file(snapshot_file)
    team_map_df = load_hasla_team_map(
        map_file,
        pd.concat([df["team"], df["opponent"]], ignore_index=True).dropna().unique(),
    )
    if snapshots_df.empty or team_map_df.empty:
        return df

    print("   -> Merging Haslametrics lagged priors...")
    return add_hasla_features(df, snapshots_df, team_map_df)


def merge_womens_net_priors(df, league, paths):
    df = ensure_womens_net_feature_columns(df)
    if league != "womens":
        return df

    snapshot_file = paths.get("womens_net_snapshot_file")
    map_file = paths.get("womens_net_map_file")
    if not snapshot_file or not map_file:
        return df

    snapshots_df = load_womens_net_snapshot_file(snapshot_file)
    team_map_df = load_womens_net_team_map(
        map_file,
        pd.concat([df["team"], df["opponent"]], ignore_index=True).dropna().unique(),
    )
    if snapshots_df.empty or team_map_df.empty:
        return df

    print("   -> Merging NCAA women NET lagged priors...")
    return add_womens_net_features(df, snapshots_df, team_map_df)


def merge_archived_market_spreads(df, league, paths):
    archive_file = paths.get("odds_archive_file")
    archived = load_latest_market_spreads(archive_file, league=league)
    if archived.empty:
        return df

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    archived["date"] = pd.to_datetime(archived["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    mirrored = pd.concat(
        [
            archived.assign(team=archived["home_team"], opponent=archived["away_team"], is_home=1),
            archived.assign(team=archived["away_team"], opponent=archived["home_team"], is_home=0, spread=-archived["spread"]),
        ],
        ignore_index=True,
    )
    mirrored = mirrored[["date", "team", "opponent", "is_home", "spread"]].rename(columns={"spread": "archived_spread"})

    merged = working.merge(mirrored, on=["date", "team", "opponent", "is_home"], how="left")
    current_spread = pd.to_numeric(merged["spread"], errors="coerce")
    archived_spread = pd.to_numeric(merged["archived_spread"], errors="coerce")

    if league == "womens":
        needs_fill = current_spread.isna() | current_spread.eq(0)
    else:
        needs_fill = current_spread.isna()

    filled_rows = int((needs_fill & archived_spread.notna()).sum())
    merged["spread"] = current_spread.where(~needs_fill, archived_spread)
    merged = merged.drop(columns=["archived_spread"])

    if filled_rows:
        print(f"   -> Filled {filled_rows} spreads from archived market data...")

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.normalize()
    return merged

def main(league="mens"):
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]

    print(f"--- FEATURE ENGINEERING (HONEST MODE: FIXED, {league}) ---")
    if not os.path.exists(data_file):
        print("❌ No data file found."); return

    # Suppress Mixed Type Warning
    df = pd.read_csv(data_file, low_memory=False)
    
    # Normalize Dates
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    
    df = clean_stale_data(df)
    
    cols = ['team_score', 'opp_score', 'spread']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')

    df = merge_archived_market_spreads(df, league, paths)
    
    df, stat_cols = calculate_advanced_stats(df)
    df = calculate_rolling_stats(df, stat_cols)
    df['ats_win'] = (df['team_score'] + df['spread'] > df['opp_score']).astype(int)
    
    df_final = merge_opponent_stats(df)
    df_final = merge_torvik_priors(df_final, league, paths)
    df_final = merge_hasla_priors(df_final, league, paths)
    df_final = merge_womens_net_priors(df_final, league, paths)

    # Neutral site: ensure column exists (0 for legacy rows without it)
    if 'is_neutral' not in df_final.columns:
        df_final['is_neutral'] = 0
    df_final['is_neutral'] = df_final['is_neutral'].fillna(0).astype(int)

    # Distance advantage from venue data
    if 'venue_city' in df_final.columns and df_final['venue_city'].notna().any():
        print("   -> Computing distance advantage...")
        geo_cache = load_geocode_cache()
        team_homes = build_team_home_locations(df_final, league=league)
        df_final = compute_distance_advantage_bulk(df_final, team_homes, geo_cache)
        save_geocode_cache(geo_cache)
    else:
        df_final['distance_advantage'] = 0.0

    print(f"Saving processed data ({len(df_final)} rows)...")
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
