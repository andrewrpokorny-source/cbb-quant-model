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


def get_last_recorded_date(data_file):
    if not os.path.exists(data_file):
        return datetime(2025, 11, 4)
    try:
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].max().to_pydatetime()
    except Exception:
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
            except: pass

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
        except Exception:
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
            df_new = pd.DataFrame(new_games)
            
            # Combine
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # NORMALIZE AND DEDUPLICATE
            df_combined['date'] = pd.to_datetime(df_combined['date']).dt.strftime('%Y-%m-%d')
            df_combined = df_combined.drop_duplicates(subset=['date', 'team'], keep='last')
            df_combined = df_combined.sort_values('date')
            
            df_combined.to_csv(data_file, index=False)
            print("✅ Database updated.")
        else:
            pd.DataFrame(new_games).to_csv(data_file, index=False)

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
