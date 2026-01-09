import pandas as pd
import requests
import os
import sys

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?limit=1000&groups=50"

def main():
    print("--- ✂️ STRING-BASED CLEANUP OF JAN 7 ---")
    
    if not os.path.exists(DATA_FILE):
        print("❌ No data file found."); return
        
    df = pd.read_csv(DATA_FILE)
    print(f"📊 Total Rows: {len(df)}")
    
    # 1. DELETE BY STRING MATCHING (The "Dumb" but Safe Way)
    # We convert the date column to string and look for the substring "2026-01-07"
    # This catches "2026-01-07", "2026-01-07 19:00:00", etc.
    
    mask = df['date'].astype(str).str.contains("2026-01-07")
    zombies = df[mask]
    
    print(f"🧟 Found {len(zombies)} rows matching '2026-01-07'.")
    if len(zombies) > 0:
        print(f"   -> Example Date Found: {zombies.iloc[0]['date']}")
    
    # Invert mask to keep everything NOT matching Jan 7
    df_clean = df[~mask].copy()
    print(f"🧹 Rows remaining: {len(df_clean)}")
    
    # 2. DOWNLOAD FRESH
    print("⬇️  Downloading fresh Jan 7 slate...")
    # We ask ESPN for 20260107
    url = f"{BASE_URL}&dates=20260107"
    try:
        res = requests.get(url).json()
    except:
        print("❌ API Connection Failed"); return

    new_games = []
    for event in res.get('events', []):
        if event['status']['type']['state'] != 'post': continue 
        
        try:
            comp = event['competitions'][0]
            home = comp['competitors'][0]
            away = comp['competitors'][1]
            
            # HARDCODED DATE STRING to ensure perfect match next time
            game_date_str = "2026-01-07"

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
            new_games.append(g)
            
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
            new_games.append(g_away)
        except: continue
        
    print(f"💾 Downloaded {len(new_games)} fresh games.")
    
    # 3. SAVE & RESTART
    if new_games:
        df_new = pd.DataFrame(new_games)
        df_final = pd.concat([df_clean, df_new], ignore_index=True)
        # Sort by date string to keep it tidy
        df_final = df_final.sort_values('date')
        
        df_final.to_csv(DATA_FILE, index=False)
        print("✅ SUCCESS: Database repaired.")
        
        print("\n--- 🚀 RESTARTING PIPELINE ---")
        os.system("python3 features.py")
        os.system("python3 backtest.py")
    else:
        print("⚠️  No games found from API. Aborting save to protect data.")

if __name__ == "__main__":
    main()