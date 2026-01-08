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
    date_str = target_date.strftime("%Y%m%d")
    print(f"   -> 📥 Downloading {target_date.strftime('%Y-%m-%d')} (Looking for Spreads)...")
    
    url = f"{BASE_URL}&dates={date_str}"
    try:
        res = requests.get(url).json()
    except:
        print(f"      ⚠️  Connection failed for {date_str}")
        return []

    games = []
    for event in res.get('events', []):
        if event['status']['type']['state'] != 'post': continue 
        
        try:
            comp = event['competitions'][0]
            home = comp['competitors'][0]
            away = comp['competitors'][1]
            
            # --- ODDS PARSING (THE FIX) ---
            spread_val = 0.0
            try:
                if comp.get('odds'):
                    odds = comp['odds'][0]
                    details = odds.get('details', '0') # e.g. "DUKE -5.5"
                    
                    if details and details != '0' and details != 'EVEN':
                        parts = details.split()
                        val = abs(float(parts[-1]))
                        fav = " ".join(parts[:-1])
                        
                        home_abbr = home['team'].get('abbreviation', '')
                        home_name = home['team'].get('displayName', '')
                        
                        is_home_fav = (fav == home_abbr) or (fav == home_name) or (fav in home_name)
                        
                        # Spread is always from Home Perspective in our DB
                        if is_home_fav:
                            spread_val = -val # Home Fav (-5.5)
                        else:
                            spread_val = val  # Home Dog (+5.5)
            except:
                spread_val = 0.0
            
            # Home Row
            g = {
                'date': target_date.strftime("%Y-%m-%d"),
                'team': home['team']['displayName'],
                'opponent': away['team']['displayName'],
                'location': 'Home',
                'team_score': int(home['score']),
                'opp_score': int(away['score']),
                'is_home': 1,
                'spread': spread_val, # SAVING THE SPREAD
                'ats_win': 0 
            }
            games.append(g)
            
            # Away Row (Inverse Spread)
            g_away = {
                'date': target_date.strftime("%Y-%m-%d"),
                'team': away['team']['displayName'],
                'opponent': home['team']['displayName'],
                'location': 'Away',
                'team_score': int(away['score']),
                'opp_score': int(home['score']),
                'is_home': 0,
                'spread': -1 * spread_val, # INVERSE SPREAD
                'ats_win': 0
            }
            games.append(g_away)
            
        except: continue
        
    return games

def update_database():
    print("--- 🔄 SMART AUTO-UPDATER (WITH ODDS FIX) ---")
    
    # FORCE RE-RUN: We purposely set the start date back to Jan 2 
    # to overwrite the "bad" data we just downloaded.
    start_date = datetime(2026, 1, 2) 
    end_date = datetime.now() - timedelta(days=1)
    
    current_date = start_date
    new_games = []
    
    print(f"📉 Repairing Data Gap: {current_date.date()} to {end_date.date()}")
    
    while current_date.date() <= end_date.date():
        daily_games = fetch_games_for_date(current_date)
        new_games.extend(daily_games)
        current_date += timedelta(days=1)
        
    if new_games:
        print(f"💾 Saving {len(new_games)} repaired records...")
        
        if os.path.exists(DATA_FILE):
            df_old = pd.read_csv(DATA_FILE)
            df_old['date'] = pd.to_datetime(df_old['date'])
            
            # DELETE the bad rows from Jan 2 onwards so we don't have duplicates
            cutoff = pd.Timestamp("2026-01-02")
            df_clean = df_old[df_old['date'] < cutoff].copy()
            
            df_new = pd.DataFrame(new_games)
            df_new['date'] = pd.to_datetime(df_new['date'])
            
            df_combined = pd.concat([df_clean, df_new], ignore_index=True)
            df_combined = df_combined.sort_values('date')
            
            df_combined.to_csv(DATA_FILE, index=False)
            print("✅ Database repaired successfully.")
        else:
            pd.DataFrame(new_games).to_csv(DATA_FILE, index=False)

    run_pipeline()

def run_pipeline():
    print("\n--- 🚀 TRIGGERING PIPELINE ---")
    print("1️⃣  Calculating Efficiency Stats...")
    os.system("python3 features.py")

    print("2️⃣  Retraining Model...")
    os.system("python3 model.py")
        
    print("3️⃣  Generating Picks...")
    os.system("python3 predict.py")

if __name__ == "__main__":
    update_database()