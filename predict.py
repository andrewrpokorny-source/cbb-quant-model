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

# --- EXPANDED TEAM MAP (V2.9: MID-MAJOR FIX) ---
TEAM_MAP = {
    # NEW ADDITIONS (Jan 7 Slate)
    "St. Thomas-Minnesota Tommies": "St. Thomas MN",
    "Elon Phoenix": "Elon",
    "Campbell Fighting Camels": "Campbell",
    "App State Mountaineers": "Appalachian St.",
    "UT Martin Skyhawks": "UT Martin",
    "Denver Pioneers": "Denver",
    "Omaha Mavericks": "Omaha", # Sometimes "Neb. Omaha" or "UNO"
    "Texas State Bobcats": "Texas St.",
    "Weber State Wildcats": "Weber St.",
    "Idaho State Bengals": "Idaho St.",
    "Louisiana Ragin' Cajuns": "Louisiana",
    "UL Monroe Warhawks": "UL Monroe",
    "Idaho Vandals": "Idaho",
    "Montana Grizzlies": "Montana",
    "UTEP Miners": "UTEP",
    "California Baptist Lancers": "California Baptist",
    "Utah Tech Trailblazers": "Utah Tech",
    "Drake Bulldogs": "Drake",
    "Butler Bulldogs": "Butler",
    "Miami (OH) RedHawks": "Miami OH",
    "Ball State Cardinals": "Ball St.",
    "Kent State Golden Flashes": "Kent St.",
    "Iowa Hawkeyes": "Iowa",
    "DePaul Blue Demons": "DePaul",
    "UNLV Rebels": "UNLV",
    "Fresno State Bulldogs": "Fresno St.",
    "Nevada Wolf Pack": "Nevada",
    "Furman Paladins": "Furman",
    "Marquette Golden Eagles": "Marquette",
    "Xavier Musketeers": "Xavier",
    "UAB Blazers": "UAB",
    "Arkansas Razorbacks": "Arkansas",
    "Duke Blue Devils": "Duke",
    "Louisville Cardinals": "Louisville",
    "North Carolina Tar Heels": "North Carolina",
    "Kentucky Wildcats": "Kentucky",
    "Kansas Jayhawks": "Kansas",
    "Auburn Tigers": "Auburn",
    "Alabama Crimson Tide": "Alabama",
    "Tennessee Volunteers": "Tennessee",
    "UConn Huskies": "Connecticut", 
    "Ole Miss Rebels": "Mississippi", 
    "NC State Wolfpack": "North Carolina St.",
    "Miami Hurricanes": "Miami FL", 
    "USC Trojans": "USC", 
    "TCU Horned Frogs": "TCU", 
    "SMU Mustangs": "SMU",
    "VCU Rams": "VCU", 
    "LSU Tigers": "LSU", 
    "BYU Cougars": "BYU", 
    "UCF Knights": "UCF",
    "St. John's Red Storm": "St. John's", 
    "Saint Mary's Gaels": "Saint Mary's", 
    "Loyola Chicago Ramblers": "Loyola Chicago",
    "Michigan State Spartans": "Michigan St.", 
    "Ohio State Buckeyes": "Ohio St.", 
    "Iowa State Cyclones": "Iowa St.",
    "Florida State Seminoles": "Florida St.", 
    "Kansas State Wildcats": "Kansas St.", 
    "Oklahoma State Cowboys": "Oklahoma St.",
    "Oregon State Beavers": "Oregon St.", 
    "Washington State Cougars": "Washington St.", 
    "Arizona State Sun Devils": "Arizona St.",
    "Mississippi State Bulldogs": "Mississippi St.", 
    "Penn State Nittany Lions": "Penn St.", 
    "Boise State Broncos": "Boise St.",
    "San Diego State Aztecs": "San Diego St.", 
    "Utah State Aggies": "Utah St.", 
    "Colorado State Rams": "Colorado St.",
    "Michigan Wolverines": "Michigan", 
    "West Virginia Mountaineers": "West Virginia", 
    "Gonzaga Bulldogs": "Gonzaga",
    "Nebraska Cornhuskers": "Nebraska", 
    "Seattle U Redhawks": "Seattle", 
    "Stanford Cardinal": "Stanford", 
    "Massachusetts Minutemen": "Massachusetts", 
    "UMass Minutemen": "Massachusetts",
    "Pittsburgh Panthers": "Pittsburgh", 
    "Illinois Fighting Illini": "Illinois", 
    "Wisconsin Badgers": "Wisconsin",
    "Maryland Terrapins": "Maryland", 
    "Rutgers Scarlet Knights": "Rutgers", 
    "Northwestern Wildcats": "Northwestern",
    "Purdue Boilermakers": "Purdue", 
    "Indiana Hoosiers": "Indiana", 
    "Minnesota Golden Gophers": "Minnesota",
    "Texas Longhorns": "Texas",
    "Texas A&M Aggies": "Texas A&M",
    "Texas Tech Red Raiders": "Texas Tech",
    "Baylor Bears": "Baylor",
    "Houston Cougars": "Houston",
    "Virginia Cavaliers": "Virginia",
    "Virginia Tech Hokies": "Virginia Tech",
    "Clemson Tigers": "Clemson",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Wake Forest Demon Deacons": "Wake Forest",
    "Syracuse Orange": "Syracuse",
    "Boston College Eagles": "Boston College",
    "Notre Dame Fighting Irish": "Notre Dame"
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
    print("   -> 📅 Fetching schedule (With Raw Odds Check)...")
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
                
                odds = comp.get('odds', [{}])[0] if comp.get('odds') else {}
                details = odds.get('details', '0')
                raw_odds = details 
                
                spread_val = 0.0
                try:
                    if details and details != '0' and details != 'EVEN':
                        parts = details.split()
                        val = abs(float(parts[-1]))
                        fav = " ".join(parts[:-1])
                        
                        home_abbr = home_tm.get('abbreviation', '')
                        is_home_fav = (fav == home_abbr) or (fav == home) or (fav in home)
                        
                        if is_home_fav:
                            spread_val = -val
                        else:
                            spread_val = val
                except: 
                    spread_val = 0.0

                if spread_val == 0.0: continue
                
                game_id = event['id']
                if not any(g['id'] == game_id for g in games):
                    games.append({
                        'id': game_id,
                        'home_raw': home, 'away_raw': away, 
                        'spread': spread_val, 'date': game_date,
                        'raw_odds': raw_odds
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
    print("--- 🔮 PREDICTION ENGINE (V2.9: MID-MAJORS) 🔮 ---")
    try:
        model = joblib.load(MODEL_FILE)
        df_hist = pd.read_csv(DATA_FILE)
    except:
        print("❌ Critical: Run model.py first."); return

    known_teams = df_hist['team'].unique()
    team_stats = get_latest_stats(df_hist)
    schedule = fetch_schedule()
    
    print(f"   -> Found {len(schedule)} games.")
    predictions = []
    
    for g in schedule:
        home = find_best_match(g['home_raw'], known_teams)
        away = find_best_match(g['away_raw'], known_teams)
        
        # DEBUG: Print if we still can't find a team
        if not home: print(f"      ⚠️  Could not map home team: {g['home_raw']}")
        if not away: print(f"      ⚠️  Could not map away team: {g['away_raw']}")
        
        if not home or not away or home not in team_stats or away not in team_stats: continue

        row = {'is_home': 1, 'spread': g['spread']}
        h_stats = team_stats[home]; a_stats = team_stats[away]
        
        last_date = pd.to_datetime(h_stats.get('last_game_date', datetime.now()))
        rest = (g['date'].replace(tzinfo=None) - last_date).days
        row['rest_days'] = min(rest, 7)
        
        row = calculate_production_features(row, h_stats, a_stats)
        
        cols = model.feature_names_in_
        input_df = pd.DataFrame([row])
        for c in cols:
            if c not in input_df.columns: input_df[c] = 0.0
        
        input_df.columns = input_df.columns.astype(str)
        prob = model.predict_proba(input_df)[0][1]
        conf = max(prob, 1-prob)
        
        if prob > 0.5:
            sign = "+" if g['spread'] > 0 else ""
            pick_str = f"{home} {sign}{g['spread']}"
        else:
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
            "Raw Odds": g['raw_odds'],
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