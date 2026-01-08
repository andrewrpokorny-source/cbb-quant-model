import pandas as pd
import requests
import os
import sys
from datetime import datetime, timedelta

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# We update the raw file, features.py handles the processed one
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv") 
BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?limit=1000&groups=50"

def get_last_recorded_date():
    """Finds the last date we have data for."""
    if not os.path.exists(DATA_FILE):
        # Default start date if file is missing (Start of season approx)
        return datetime(2025, 11, 4) 
    
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        return df['date'].max().to_pydatetime()
    except:
        return datetime(2025, 11, 4)

def fetch_games_for_date(target_date):
    """Fetches a single day of games from ESPN."""
    date_str = target_date.strftime("%Y%m%d")
    print(f"   -> 📥 Downloading {target_date.strftime('%Y-%m-%d')}...")
    
    url = f"{BASE_URL}&dates={date_str}"
    try:
        res = requests.get(url).json()
    except:
        print(f"      ⚠️  Connection failed for {date_str}")
        return []

    games = []
    for event in res.get('events', []):
        if event['status']['type']['state'] != 'post': continue # Only finished games
        
        try:
            comp = event['competitions'][0]
            home = comp['competitors'][0]
            away = comp['competitors'][1]
            
            # Extract basic data needed for features.py
            g = {
                'date': target_date.strftime("%Y-%m-%d"),
                'team': home['team']['displayName'],
                'opponent': away['team']['displayName'],
                'location': 'Home',
                'team_score': int(home['score']),
                'opp_score': int(away['score']),
                'is_home': 1,
                'ats_win': 0 # Placeholder, calculated later
            }
            # We add the "Home" perspective
            games.append(g)
            
            # We add the "Away" perspective
            g_away = {
                'date': target_date.strftime("%Y-%m-%d"),
                'team': away['team']['displayName'],
                'opponent': home['team']['displayName'],
                'location': 'Away',
                'team_score': int(away['score']),
                'opp_score': int(home['score']),
                'is_home': 0,
                'ats_win': 0
            }
            games.append(g_away)
            
        except: continue
        
    return games

def update_database():
    print("--- 🔄 SMART AUTO-UPDATER ---")
    
    last_date = get_last_recorded_date()
    # Start checking from the day AFTER our last record
    current_date = last_date + timedelta(days=1)
    # Stop at Yesterday (since today's games aren't done)
    end_date = datetime.now() - timedelta(days=1)
    
    # Correction: If we are running this late at night, we might want today's early games
    # But usually safe to stick to "Yesterday" to ensure final scores.
    
    if current_date.date() > end_date.date():
        print("✅ Data is already up to date!")
        run_pipeline()
        return

    print(f"📉 Found Data Gap: {current_date.date()} to {end_date.date()}")
    
    new_games = []
    while current_date.date() <= end_date.date():
        daily_games = fetch_games_for_date(current_date)
        new_games.extend(daily_games)
        current_date += timedelta(days=1)
        
    if new_games:
        print(f"💾 Saving {len(new_games)} new game records...")
        
        # Load existing
        if os.path.exists(DATA_FILE):
            df_old = pd.read_csv(DATA_FILE)
            df_new = pd.DataFrame(new_games)
            # Combine and remove duplicates
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined['date'] = pd.to_datetime(df_combined['date'])
            df_combined = df_combined.drop_duplicates(subset=['date', 'team'])
            df_combined = df_combined.sort_values('date')
        else:
            df_combined = pd.DataFrame(new_games)
            
        df_combined.to_csv(DATA_FILE, index=False)
        print("✅ Database updated successfully.")
    else:
        print("⚠️  No new games found in gap (Off-season or API issue?).")

    run_pipeline()

def run_pipeline():
    print("\n--- 🚀 TRIGGERING PIPELINE ---")
    # This runs the rest of your system automatically
    
    print("1️⃣  Calculating Efficiency Stats (features.py)...")
    if os.system("python3 features.py") != 0:
        print("❌ Error in features.py"); return

    print("2️⃣  Retraining Model (model.py)...")
    if os.system("python3 model.py") != 0:
        print("❌ Error in model.py"); return
        
    print("3️⃣  Generating Picks (predict.py)...")
    os.system("python3 predict.py")

if __name__ == "__main__":
    update_database()