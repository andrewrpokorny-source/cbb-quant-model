import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

from league_config import (
    get_league_artifact_paths,
    get_league_settings,
    get_scoreboard_base_url,
    normalize_league,
)

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENUE_METADATA_COLUMNS = ["is_neutral", "venue_city", "venue_state"]


def ensure_venue_columns(df):
    """Ensure venue-related columns exist with stable defaults."""
    if "is_neutral" not in df.columns:
        df["is_neutral"] = 0
    else:
        df["is_neutral"] = pd.to_numeric(df["is_neutral"], errors="coerce").fillna(0).astype(int)

    for col in ("venue_city", "venue_state"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)
    return df


def merge_venue_metadata(existing_df, metadata_df):
    """Backfill venue metadata for existing rows using date/team matches."""
    if metadata_df.empty:
        return ensure_venue_columns(existing_df)

    existing_df = ensure_venue_columns(existing_df.copy())
    metadata_df = ensure_venue_columns(metadata_df.copy())

    meta_cols = ["date", "team"] + VENUE_METADATA_COLUMNS
    merged = existing_df.merge(
        metadata_df[meta_cols].drop_duplicates(subset=["date", "team"], keep="last"),
        on=["date", "team"],
        how="left",
        suffixes=("", "_new"),
    )

    merged["is_neutral"] = merged["is_neutral"].where(
        merged["is_neutral"] != 0,
        merged["is_neutral_new"].fillna(0),
    ).fillna(0).astype(int)

    for col in ("venue_city", "venue_state"):
        merged[col] = merged[col].where(
            merged[col].astype(str).str.strip() != "",
            merged[f"{col}_new"].fillna(""),
        )

    drop_cols = [f"{col}_new" for col in VENUE_METADATA_COLUMNS]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged


def backfill_venue_metadata(df, base_url):
    """Fetch and merge missing venue metadata for historical rows."""
    df = ensure_venue_columns(df.copy())
    missing_mask = (
        (df["is_neutral"] == 0)
        & (df["venue_city"].astype(str).str.strip() == "")
        & (df["venue_state"].astype(str).str.strip() == "")
    )
    target_dates = sorted(pd.to_datetime(df.loc[missing_mask, "date"]).dt.normalize().dropna().unique())
    if not target_dates:
        return df

    print(f"🧭 Backfilling venue metadata for {len(target_dates)} dates...")
    fetched = []
    for idx, ts in enumerate(target_dates, start=1):
        fetched.extend(fetch_games_for_date(pd.Timestamp(ts).to_pydatetime(), base_url))
        if idx % 30 == 0 or idx == len(target_dates):
            print(f"   -> Venue backfill {idx}/{len(target_dates)} dates")

    if not fetched:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    fetched_df = pd.DataFrame(fetched)
    fetched_df["date"] = pd.to_datetime(fetched_df["date"]).dt.strftime("%Y-%m-%d")
    return merge_venue_metadata(df, fetched_df)


def get_last_recorded_date(data_file):
    if not os.path.exists(data_file):
        return datetime(2025, 11, 4)
    try:
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].max().to_pydatetime()
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        print(f"WARNING: Data file corrupted ({e}), will re-download from season start")
        return datetime(2025, 11, 4)
    except Exception as e:
        print(f"WARNING: Could not read last date from {data_file} ({type(e).__name__}: {e})")
        return datetime(2025, 11, 4)


def fetch_games_for_date(target_date, base_url):
    date_str_url = target_date.strftime("%Y%m%d")
    print(f"   -> 📥 Downloading {target_date.strftime('%Y-%m-%d')}...")
    
    url = f"{base_url}&dates={date_str_url}"
    try:
        res = requests.get(url).json()
    except Exception:
        print(f"      ⚠️  Connection failed for {date_str_url}")
        return []

    games = []
    for event in res.get('events', []):
        if event['status']['type']['state'] != 'post': continue 
        
        try:
            comp = event['competitions'][0]
            home = comp['competitors'][0]
            away = comp['competitors'][1]
            
            # NORMALIZE DATE (No Time)
            game_date_str = target_date.strftime("%Y-%m-%d")

            spread_val = 0.0
            try:
                if comp.get('odds'):
                    odds = comp['odds'][0]
                    details = odds.get('details', '0')
                    if details and details != '0' and details != 'EVEN':
                        parts = details.split()
                        val = abs(float(parts[-1]))
                        fav = " ".join(parts[:-1])
                        home_abbr = home['team'].get('abbreviation', '')
                        home_name = home['team'].get('displayName', '')
                        is_home_fav = (fav == home_abbr) or (fav == home_name) or (fav in home_name)
                        spread_val = -val if is_home_fav else val
            except (ValueError, IndexError, TypeError):
                pass

            # Neutral site + venue info
            is_neutral = int(comp.get('neutralSite', False))
            venue = comp.get('venue', {})
            venue_addr = venue.get('address', {})
            venue_city = venue_addr.get('city', '')
            venue_state = venue_addr.get('state', '')

            g = {
                'date': game_date_str,
                'team': home['team']['displayName'],
                'opponent': away['team']['displayName'],
                'location': 'Home',
                'team_score': int(home['score']),
                'opp_score': int(away['score']),
                'is_home': 1,
                'spread': spread_val,
                'is_neutral': is_neutral,
                'venue_city': venue_city,
                'venue_state': venue_state,
                'ats_win': 0
            }
            games.append(g)

            g_away = {
                'date': game_date_str,
                'team': away['team']['displayName'],
                'opponent': home['team']['displayName'],
                'location': 'Away',
                'team_score': int(away['score']),
                'opp_score': int(home['score']),
                'is_home': 0,
                'spread': -1 * spread_val,
                'is_neutral': is_neutral,
                'venue_city': venue_city,
                'venue_state': venue_state,
                'ats_win': 0
            }
            games.append(g_away)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            event_id = event.get('id', 'unknown')
            print(f"      Skipped event {event_id}: {type(e).__name__}: {e}")
            continue
        
    return games


def update_database(league="mens"):
    league = normalize_league(league)
    settings = get_league_settings(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    base_url = get_scoreboard_base_url(league)

    print(f"--- 🔄 AUTO-HEALING UPDATER ({settings['label']}) ---")
    
    # 1. Determine Range
    last_date = get_last_recorded_date(data_file)
    # Start from the day AFTER the last record
    start_date = last_date + timedelta(days=1)
    
    # End Yesterday (Strictly ignore Today to avoid partial games)
    end_date = datetime.now() - timedelta(days=1)
    
    # If the database is somehow ahead of reality (timezones), cap it
    if start_date.date() > end_date.date():
        print(f"✅ Data is up to date! (Last: {last_date.date()})")
        if os.path.exists(data_file):
            df_current = pd.read_csv(data_file)
            df_backfilled = backfill_venue_metadata(df_current, base_url)
            if not df_backfilled.equals(df_current):
                df_backfilled.to_csv(data_file, index=False)
                print("✅ Venue metadata backfilled.")
        run_pipeline(league)
        return

    print(f"📉 Filling Gap: {start_date.date()} to {end_date.date()}")
    
    new_games = []
    current_date = start_date
    while current_date.date() <= end_date.date():
        daily_games = fetch_games_for_date(current_date, base_url)
        new_games.extend(daily_games)
        current_date += timedelta(days=1)
        
    if new_games:
        print(f"💾 Saving {len(new_games)} new games...")
        
        if os.path.exists(data_file):
            df_old = pd.read_csv(data_file)
            df_old = ensure_venue_columns(df_old)
            df_new = pd.DataFrame(new_games)
            df_new = ensure_venue_columns(df_new)
            
            # Combine
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # NORMALIZE AND DEDUPLICATE
            df_combined['date'] = pd.to_datetime(df_combined['date']).dt.strftime('%Y-%m-%d')
            df_combined = df_combined.drop_duplicates(subset=['date', 'team'], keep='last')
            df_combined = df_combined.sort_values('date')
            
            df_combined.to_csv(data_file, index=False)
            print("✅ Database updated.")
        else:
            pd.DataFrame(new_games).pipe(ensure_venue_columns).to_csv(data_file, index=False)

    if os.path.exists(data_file):
        df_current = pd.read_csv(data_file)
        df_backfilled = backfill_venue_metadata(df_current, base_url)
        if not df_backfilled.equals(df_current):
            df_backfilled.to_csv(data_file, index=False)
            print("✅ Venue metadata backfilled.")

    run_pipeline(league)


def run_pipeline(league):
    print("\n--- 🚀 TRIGGERING PIPELINE ---")
    print("1️⃣  Calculating Efficiency Stats...")
    subprocess.run([sys.executable, "features.py", "--league", league], check=False)

    print("2️⃣  Grading History...")
    subprocess.run([sys.executable, "backtest.py", "--league", league], check=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update CBB data and run feature/backtest pipeline.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League to update: mens or womens (aliases: men/women/cbb/wcbb).",
    )
    args = parser.parse_args()
    update_database(args.league)
