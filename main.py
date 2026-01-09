import pandas as pd
import requests
import os
import sys
from datetime import datetime, timedelta

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv") 
BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?limit=1000&groups=50"

def get_last_recorded_date():
    if not os.path.exists(DATA_FILE):
        return datetime(2025, 11, 4) 
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].max().to_pydatetime()
    except:
        return datetime(2025, 11, 4)

def fetch_games_for_date(target_date):
    date_str_url = target_date.strftime("%Y%m%d")
    print(f"   -> 📥 Downloading {target_date.strftime('%Y-%m-%d')}...")
    
    url = f"{BASE_URL}&dates={date_str_url}"
    try:
        res = requests.get(url).json()
    except:
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
            
            g = {
                'date': game_date_str,
                'team': home['team']['displayName'],
                'opponent': away['team']['displayName'],
                'location': 'Home',
                'team_score': int(home['score']),
                'opp_score': int(away['score']),
                'is_home': 1,
                'spread': spread_val,
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
                'ats_win': 0
            }
            games.append(g_away)
        except: continue
        
    return games

def update_database():
    print("--- 🔄 AUTO-HEALING UPDATER ---")
    
    # 1. Determine Range
    last_date = get_last_recorded_date()
    # Start from the day AFTER the last record
    start_date = last_date + timedelta(days=1)
    
    # End Yesterday (Strictly ignore Today to avoid partial games)
    end_date = datetime.now() - timedelta(days=1)
    
    # If the database is somehow ahead of reality (timezones), cap it
    if start_date.date() > end_date.date():
        print(f"✅ Data is up to date! (Last: {last_date.date()})")
        run_pipeline()
        return

    print(f"📉 Filling Gap: {start_date.date()} to {end_date.date()}")
    
    new_games = []
    current_date = start_date
    while current_date.date() <= end_date.date():
        daily_games = fetch_games_for_date(current_date)
        new_games.extend(daily_games)
        current_date += timedelta(days=1)
        
    if new_games:
        print(f"💾 Saving {len(new_games)} new games...")
        
        if os.path.exists(DATA_FILE):
            df_old = pd.read_csv(DATA_FILE)
            df_new = pd.DataFrame(new_games)
            
            # Combine
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            
            # NORMALIZE AND DEDUPLICATE
            df_combined['date'] = pd.to_datetime(df_combined['date']).dt.strftime('%Y-%m-%d')
            df_combined = df_combined.drop_duplicates(subset=['date', 'team'], keep='last')
            df_combined = df_combined.sort_values('date')
            
            df_combined.to_csv(DATA_FILE, index=False)
            print("✅ Database updated.")
        else:
            pd.DataFrame(new_games).to_csv(DATA_FILE, index=False)

    run_pipeline()

def run_pipeline():
    print("\n--- 🚀 TRIGGERING PIPELINE ---")
    print("1️⃣  Calculating Efficiency Stats...")
    os.system("python3 features.py")

    print("2️⃣  Grading History...")
    os.system("python3 backtest.py")

if __name__ == "__main__":
    update_database()