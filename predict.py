import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timedelta
from difflib import get_close_matches

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "cbb_model_v1.pkl")
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "daily_predictions.csv")

BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=1000"

TEAM_MAP = {
    "UConn Huskies": "Connecticut", "Ole Miss Rebels": "Mississippi", "NC State Wolfpack": "North Carolina St.",
    "Miami Hurricanes": "Miami FL", "USC Trojans": "USC", "TCU Horned Frogs": "TCU", "SMU Mustangs": "SMU",
    "VCU Rams": "VCU", "LSU Tigers": "LSU", "BYU Cougars": "BYU", "UCF Knights": "UCF",
    "St. John's Red Storm": "St. John's", "Saint Mary's Gaels": "Saint Mary's", "Loyola Chicago Ramblers": "Loyola Chicago",
    "Michigan State Spartans": "Michigan St.", "Ohio State Buckeyes": "Ohio St.", "Iowa State Cyclones": "Iowa St.",
    "Florida State Seminoles": "Florida St.", "Kansas State Wildcats": "Kansas St.", "Oklahoma State Cowboys": "Oklahoma St.",
    "Oregon State Beavers": "Oregon St.", "Washington State Cougars": "Washington St.", "Arizona State Sun Devils": "Arizona St.",
    "Mississippi State Bulldogs": "Mississippi St.", "Penn State Nittany Lions": "Penn St.", "Boise State Broncos": "Boise St.",
    "San Diego State Aztecs": "San Diego St.", "Utah State Aggies": "Utah St.", "Colorado State Rams": "Colorado St.",
    "Michigan Wolverines": "Michigan", "West Virginia Mountaineers": "West Virginia", "Gonzaga Bulldogs": "Gonzaga",
    "Nebraska Cornhuskers": "Nebraska", "Seattle U Redhawks": "Seattle", "Louisville Cardinals": "Louisville",
    "Stanford Cardinal": "Stanford", "Massachusetts Minutemen": "Massachusetts", "UMass Minutemen": "Massachusetts",
    "Pittsburgh Panthers": "Pittsburgh", "Illinois Fighting Illini": "Illinois", "Wisconsin Badgers": "Wisconsin",
    "Maryland Terrapins": "Maryland", "Rutgers Scarlet Knights": "Rutgers", "Northwestern Wildcats": "Northwestern",
    "Purdue Boilermakers": "Purdue", "Indiana Hoosiers": "Indiana", "Minnesota Golden Gophers": "Minnesota"
}

def find_best_match(name, known_teams):
    if name in TEAM_MAP: return TEAM_MAP[name]
    parts = name.split()
    if len(parts) > 1:
        no_mascot = " ".join(parts[:-1])
        if no_mascot in TEAM_MAP: return TEAM_MAP[no_mascot]
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_latest_stats(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    latest_stats = {}
    teams = df['team'].unique()
    for team in teams:
        last_game = df[df['team'] == team].iloc[-1]
        stats = {}
        for col in df.columns:
            if any(x in col for x in ['season_', 'roll']):
                stats[col] = last_game[col]
        stats['last_game_date'] = last_game['date']
        latest_stats[team] = stats
    return latest_stats

def fetch_schedule():
    print("   -> 📅 Fetching schedule (48-Hour Lookahead)...")
    games = []
    
    for days_ahead in [0, 1]:
        target_date = datetime.now() + timedelta(days=days_ahead)
        date_str = target_date.strftime("%Y%m%d")
        url = f"{BASE_URL}&dates={date_str}"
        
        try:
            res = requests.get(url)
            data = res.json()
            
            for event in data['events']:
                comp = event['competitions'][0]
                status = event['status']['type']['state']
                if status == 'post': continue 
                
                game_date = pd.to_datetime(event['date'])
                
                if not comp.get('competitors'): continue
                home_tm = comp['competitors'][0]['team']
                away_tm = comp['competitors'][1]['team']
                home = home_tm['displayName']; away = away_tm['displayName']
                
                # PARSING ODDS
                odds = comp.get('odds', [{}])[0] if comp.get('odds') else {}
                details = odds.get('details', '0')
                spread_val = 0.0
                
                try:
                    if details and details != '0' and details != 'EVEN':
                        parts = details.split()
                        val = float(parts[-1]) # e.g., 18.5
                        fav = " ".join(parts[:-1]) # e.g., "Vandy"
                        
                        # Identify who is favored
                        home_abbr = home_tm.get('abbreviation', '')
                        
                        # Logic: Spread is always from Home perspective
                        # If Home is favored: Spread is Negative (e.g., -18.5)
                        # If Away is favored: Spread is Positive (e.g., +18.5)
                        
                        if fav == home_abbr or fav == home or fav in home:
                            spread_val = -val
                        else:
                            spread_val = val
                except: 
                    spread_val = 0.0

                # FILTER: Skip games with 0.0 spread (No line posted yet)
                if spread_val == 0.0:
                    continue
                
                game_id = event['id']
                if not any(g['id'] == game_id for g in games):
                    games.append({
                        'id': game_id,
                        'home_raw': home, 'away_raw': away, 
                        'spread': spread_val, 'date': game_date
                    })
        except: pass
            
    return sorted(games, key=lambda x: x['date'])

def calculate_production_features(row, h_stats, a_stats):
    row['diff_eFG'] = h_stats.get('season_team_eFG', 0) - a_stats.get('season_team_eFG', 0)
    row['diff_Rebound'] = h_stats.get('season_team_ORB', 0) - a_stats.get('season_team_ORB', 0)
    row['diff_TO'] = h_stats.get('season_team_TO', 0) - a_stats.get('season_team_TO', 0)
    row['momentum_gap'] = h_stats.get('roll3_team_eFG', 0) - h_stats.get('season_team_eFG', 0)
    row['roll5_cover_margin'] = h_stats.get('roll5_cover_margin', 0)
    return row

def main():
    print("--- 🔮 PREDICTION ENGINE (SMART FILTER) 🔮 ---")
    try:
        model = joblib.load(MODEL_FILE)
        df_hist = pd.read_csv(DATA_FILE)
    except:
        print("❌ Critical: Run model.py first."); return

    known_teams = df_hist['team'].unique()
    team_stats = get_latest_stats(df_hist)
    schedule = fetch_schedule()
    
    print(f"   -> Found {len(schedule)} actionable games (Lines Posted).")
    predictions = []
    
    for g in schedule:
        home = find_best_match(g['home_raw'], known_teams)
        away = find_best_match(g['away_raw'], known_teams)
        
        if not home or not away or home not in team_stats or away not in team_stats: continue

        row = {'is_home': 1, 'spread': g['spread']}
        h_stats = team_stats[home]; a_stats = team_stats[away]
        
        last_date = pd.to_datetime(h_stats.get('last_game_date', datetime.now()))
        row['rest_days'] = min((g['date'].replace(tzinfo=None) - last_date).days, 7)
        row = calculate_production_features(row, h_stats, a_stats)
        
        cols = model.feature_names_in_
        input_df = pd.DataFrame([row])
        for c in cols:
            if c not in input_df.columns: input_df[c] = 0.0
        
        input_df.columns = input_df.columns.astype(str)
        
        prob = model.predict_proba(input_df)[0][1]
        conf = max(prob, 1-prob)
        
        # --- ROBUST PICK STRING GENERATION ---
        # g['spread'] is the HOME SPREAD.
        # If g['spread'] = -18.5 (Home Favored by 18.5)
        # If g['spread'] = 18.5 (Home Underdog by 18.5, aka Away Favored by 18.5)
        
        if prob > 0.5:
            # We are picking the HOME TEAM
            # If spread is -18.5, we display "Home -18.5"
            # If spread is 18.5, we display "Home +18.5"
            sign = "+" if g['spread'] > 0 else ""
            pick_str = f"{home} {sign}{g['spread']}"
        else:
            # We are picking the AWAY TEAM
            # We need the AWAY SPREAD, which is -1 * HOME SPREAD
            away_spread = -1 * g['spread']
            sign = "+" if away_spread > 0 else ""
            pick_str = f"{away} {sign}{away_spread}"

        try:
            local_ts = g['date'].tz_convert('US/Eastern')
            time_str = local_ts.strftime("%m/%d %I:%M %p")
        except:
            time_str = g['date'].strftime("%I:%M %p")

        predictions.append({
            "Date/Time": time_str,
            "Matchup": f"{away} @ {home}",
            "Spread": g['spread'],
            "Pick": pick_str,
            "Conf": conf,
            "Rest": row['rest_days']
        })

    if predictions:
        pred_df = pd.DataFrame(predictions).sort_values(by="Conf", ascending=False)
        print("\n--- 💰 PREDICTIONS GENERATED ---")
        pred_df.to_csv(OUTPUT_FILE, index=False)
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    main()