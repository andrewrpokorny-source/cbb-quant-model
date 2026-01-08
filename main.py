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
            
            # --- TIMEZONE FIX ---
            utc_ts = pd.to_datetime(event['date'])
            eastern_ts = utc_ts.tz_convert('US/Eastern')
            game_date_str = eastern_ts.strftime("%Y-%m-%d")

            # --- ODDS PARSING ---
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
                        
                        if is_home_fav:
                            spread_val = -val
                        else:
                            spread_val = val
            except:
                spread_val = 0.0
            
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
    print("--- 🔄 STRICT UPDATER (YESTERDAY ONLY) ---")
    
    # We purposefully set start_date back to repair the gap, 
    # but end_date is strictly YESTERDAY.
    start_date = datetime(2026, 1, 2) 
    
    # STRICT CUTOFF: Yesterday Only. Ignore Today.
    end_date = datetime.now() - timedelta(days=1)
    
    current_date = start_date
    new_games = []
    
    print(f"📉 Scanning from {current_date.date()} to {end_date.date()}")
    
    while current_date.date() <= end_date.date():
        daily_games = fetch_games_for_date(current_date)
        new_games.extend(daily_games)
        current_date += timedelta(days=1)
        
    if new_games:
        print(f"💾 Found {len(new_games)} completed games.")
        
        if os.path.exists(DATA_FILE):
            df_old = pd.read_csv(DATA_FILE)
            df_old['date'] = pd.to_datetime(df_old['date'])
            
            # Clean overwrite for the rescue period
            cutoff = pd.Timestamp("2026-01-02")
            df_clean = df_old[df_old['date'] < cutoff].copy()
            
            df_new = pd.DataFrame(new_games)
            df_new['date'] = pd.to_datetime(df_new['date'])
            
            df_combined = pd.concat([df_clean, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['date', 'team'])
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

    print("2️⃣  Retraining Model...")
    os.system("python3 model.py")
        
    print("3️⃣  Generating Picks...")
    os.system("python3 predict.py")

if __name__ == "__main__":
    update_database()