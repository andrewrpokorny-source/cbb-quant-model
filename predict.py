import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime
from difflib import get_close_matches

# --- CONFIG ---
MODEL_FILE = "cbb_model_v1.pkl"
DATA_FILE = "cbb_training_data_processed.csv"
OUTPUT_FILE = "daily_predictions.csv"
# groups=50 gets ALL NCAA Division I games
SCHEDULE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=1000"

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
    print("   -> 📅 Fetching full D1 schedule...")
    try:
        res = requests.get(SCHEDULE_URL)
        data = res.json()
        games = []
        for event in data['events']:
            comp = event['competitions'][0]
            
            # Skip games that are already Final
            status = event['status']['type']['state']
            if status == 'post': continue 
            
            # Capture Real Date
            date_str = event['date'] # ISO Format (UTC)
            game_date = pd.to_datetime(date_str)

            if not comp.get('competitors'): continue
            home_tm = comp['competitors'][0]['team']
            away_tm = comp['competitors'][1]['team']
            home = home_tm['displayName']; away = away_tm['displayName']
            
            # Parsing Odds
            odds = comp.get('odds', [{}])[0] if comp.get('odds') else {}
            details = odds.get('details', '0')
            spread_val = 0.0
            try:
                if details and details != '0':
                    parts = details.split()
                    val = float(parts[-1])
                    home_abbr = home_tm.get('abbreviation', '')
                    fav = " ".join(parts[:-1])
                    
                    # --- THE FIX ---
                    # If Home is favored (val is negative), Home Spread = val
                    if fav == home_abbr or fav == home or fav in home: 
                        spread_val = val
                    # If Away is favored (val is negative), Home Spread = -val (Positive)
                    else: 
                        spread_val = -val
            except: pass
            
            games.append({
                'home_raw': home, 
                'away_raw': away, 
                'spread': spread_val, 
                'date': game_date
            })
        return games
    except: return []

def calculate_production_features(row, h_stats, a_stats):
    row['diff_eFG'] = h_stats.get('season_team_eFG', 0) - a_stats.get('season_team_eFG', 0)
    row['diff_Rebound'] = h_stats.get('season_team_ORB', 0) - a_stats.get('season_team_ORB', 0)
    row['diff_TO'] = h_stats.get('season_team_TO', 0) - a_stats.get('season_team_TO', 0)
    row['momentum_gap'] = h_stats.get('roll3_team_eFG', 0) - h_stats.get('season_team_eFG', 0)
    row['roll5_cover_margin'] = h_stats.get('roll5_cover_margin', 0)
    return row

def main():
    print("--- 🔮 PREDICTION ENGINE (ALL GAMES + TIMES) 🔮 ---")
    try:
        model = joblib.load(MODEL_FILE)
        df_hist = pd.read_csv(DATA_FILE)
    except:
        print("❌ Critical: Run model.py first."); return

    known_teams = df_hist['team'].unique()
    team_stats = get_latest_stats(df_hist)
    schedule = fetch_schedule()
    
    print(f"   -> Found {len(schedule)} upcoming games.")
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
        
        # PICK FORMATTING
        # If g['spread'] is -21.5 (Michigan Favored):
        #   If prob > 0.5 (Pick Home): "Michigan -21.5"
        #   If prob < 0.5 (Pick Away): "USC +21.5"
        
        if prob > 0.5:
            pick_str = f"{home} {g['spread']}" if g['spread'] < 0 else f"{home} +{g['spread']}"
        else:
             pick_str = f"{away} +{-1 * g['spread']}" if g['spread'] < 0 else f"{away} {-1 * g['spread']}"

        # TIME FORMATTING (Convert to ET)
        try:
            # Convert to Eastern Time (US/Eastern)
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