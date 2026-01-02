import requests
import pandas as pd
from datetime import timedelta, date
import time
import os

# --- CONFIG ---
START_DATE = date(2024, 11, 4) # Start of 24-25 Season
END_DATE = date.today()
OUTPUT_FILE = "espn_odds_history.csv"

def fetch_history():
    print(f"--- 🕰️ SPINNING UP THE TIME MACHINE ({START_DATE} to {END_DATE}) ---")
    
    all_odds = []
    current_date = START_DATE
    
    while current_date <= END_DATE:
        date_str = current_date.strftime("%Y%m%d")
        print(f"   -> Scanning {current_date}...", end="\r")
        
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={date_str}&limit=500"
        
        try:
            res = requests.get(url)
            data = res.json()
            
            for event in data.get('events', []):
                comp = event['competitions'][0]
                
                # We need completed games with Odds
                if not comp.get('odds'): continue
                
                # Teams
                home_team = comp['competitors'][0]['team']['displayName']
                away_team = comp['competitors'][1]['team']['displayName']
                
                # Odds
                odds = comp['odds'][0]
                # ESPN gives "overUnder" directly usually
                total = odds.get('overUnder')
                details = odds.get('details') # Spread string like "DUKE -5.0"
                
                if total:
                    all_odds.append({
                        "date": current_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "total_line": total,
                        "spread_details": details
                    })
                    
        except Exception as e:
            print(f"\n      ❌ Error on {date_str}: {e}")
            
        current_date += timedelta(days=1)
        # Be nice to the API
        time.sleep(0.1)

    print(f"\n✅ Scan Complete. Found {len(all_odds)} games with betting lines.")
    
    df = pd.DataFrame(all_odds)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"   -> Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_history()