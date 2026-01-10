import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timedelta
from difflib import get_close_matches
import pytz

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "cbb_model_v1.pkl")
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "daily_predictions.csv")
BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=1000"

# --- TEAM MAP ---
TEAM_MAP = {
    "St. Thomas-Minnesota Tommies": "St. Thomas MN",
    "Elon Phoenix": "Elon",
    "Campbell Fighting Camels": "Campbell",
    "App State Mountaineers": "Appalachian St.",
    "UT Martin Skyhawks": "UT Martin",
    "Denver Pioneers": "Denver",
    "Omaha Mavericks": "Omaha",
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
    "Notre Dame Fighting Irish": "Notre Dame",
    "Northeastern Huskies": "Northeastern",
    # --- Newly Added Schools (Jan 2026) ---
    "Pennsylvania Quakers": "Penn",
    "UAlbany Great Danes": "Albany NY",
    "Tulsa Golden Hurricane": "Tulsa",
    "Tulane Green Wave": "Tulane",
    "Hawai'i Rainbow Warriors": "Hawaii",
    "Wagner Seahawks": "Wagner",
    "Long Island University Sharks": "Long Island"
}

def find_best_match(name, known_teams):
    """Match ESPN team name to historical data team name."""
    if name in TEAM_MAP: 
        return TEAM_MAP[name]
    
    parts = name.split()
    if len(parts) > 1:
        no_mascot = " ".join(parts[:-1])
        if no_mascot in TEAM_MAP: 
            return TEAM_MAP[no_mascot]
    
    matches = get_close_matches(name, known_teams, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    # Log warning for unmatched teams
    print(f"      ⚠️  WARNING: Could not match '{name}' to historical data")
    return None

def get_latest_stats(df):
    """Get the most recent stats for each team."""
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
        stats['last_opponent'] = last_game.get('opponent', 'Unknown')
        latest_stats[team] = stats
    
    return latest_stats

def fetch_schedule():
    """
    Fetch today's and tomorrow's games with TIMEZONE AWARENESS.
    Uses Eastern Time to ensure we're querying the correct date.
    """
    print("   -> 📅 Fetching schedule (TIMEZONE AWARE)...")
    
    # Use Eastern Time for proper date handling
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    
    print(f"      Current Eastern Time: {now_eastern.strftime('%Y-%m-%d %I:%M %p %Z')}")
    
    games = []
    
    # Fetch today and tomorrow (Eastern time)
    for days_ahead in [0, 1]:
        target_date = now_eastern + timedelta(days=days_ahead)
        date_str = target_date.strftime("%Y%m%d")
        url = f"{BASE_URL}&dates={date_str}"
        
        print(f"      Querying ESPN for: {date_str} ({target_date.strftime('%A, %B %d')})")
        
        try:
            res = requests.get(url, timeout=10)
            data = res.json()
            
            events_count = len(data.get('events', []))
            print(f"         Found {events_count} events")
            
            for event in data['events']:
                game_date = pd.to_datetime(event['date'])
                
                if not event.get('competitions'): 
                    continue
                comp = event['competitions'][0]
                
                if not comp.get('competitors'): 
                    continue
                    
                home_tm = comp['competitors'][0]['team']
                away_tm = comp['competitors'][1]['team']
                home_raw = home_tm['displayName']
                away_raw = away_tm['displayName']
                
                # Get odds
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
                        is_home_fav = (fav == home_abbr) or (fav == home_raw) or (fav in home_raw)
                        
                        if is_home_fav:
                            spread_val = -val
                        else:
                            spread_val = val
                except: 
                    spread_val = 0.0

                # Skip games without spreads
                if spread_val == 0.0: 
                    continue
                
                game_id = event['id']
                if not any(g['id'] == game_id for g in games):
                    games.append({
                        'id': game_id,
                        'home_raw': home_raw,  # Keep original ESPN name
                        'away_raw': away_raw,  # Keep original ESPN name
                        'spread': spread_val, 
                        'date': game_date,
                        'raw_odds': raw_odds
                    })
                    
        except Exception as e:
            print(f"         ❌ Error fetching {date_str}: {e}")
            
    return sorted(games, key=lambda x: x['date'])

def calculate_production_features(row, h_stats, a_stats):
    """Calculate features needed for prediction."""
    # 1. Effective Field Goal %
    row['diff_eFG'] = h_stats.get('season_team_eFG', 0) - a_stats.get('season_team_eFG', 0)
    
    # 2. Rebounds
    h_orb = h_stats.get('season_team_orb', 0) 
    a_orb = a_stats.get('season_team_orb', 0)
    row['diff_Rebound'] = h_orb - a_orb
    
    # 3. Turnovers
    h_to = h_stats.get('season_team_to', 0)
    a_to = a_stats.get('season_team_to', 0)
    row['diff_TO'] = h_to - a_to
    
    # 4. Momentum
    row['momentum_gap'] = h_stats.get('roll3_team_eFG', 0) - h_stats.get('season_team_eFG', 0)
    
    # 5. Cover Margin
    row['roll5_cover_margin'] = h_stats.get('roll5_cover_margin', 0)
    
    return row

def main():
    print("--- 🔮 PREDICTION ENGINE (V3.3: TIMEZONE + MATCHUP FIX) 🔮 ---")
    
    # Get current Eastern time for dated file naming
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    
    # Load model and data
    try:
        model = joblib.load(MODEL_FILE)
        print(f"   ✅ Model loaded: {MODEL_FILE}")
    except:
        print("❌ Critical: Model not found. Run model.py first.")
        return
    
    try:
        df_hist = pd.read_csv(DATA_FILE)
        print(f"   ✅ Data loaded: {len(df_hist)} historical games")
    except:
        print("❌ Critical: Training data not found. Run main.py to download data.")
        return

    known_teams = df_hist['team'].unique()
    team_stats = get_latest_stats(df_hist)
    
    # Check data freshness
    df_hist['date'] = pd.to_datetime(df_hist['date'])
    last_data_date = df_hist['date'].max()
    print(f"   📊 Data current through: {last_data_date.strftime('%Y-%m-%d')}")
    
    days_old = (datetime.now() - last_data_date).days
    if days_old > 2:
        print(f"   ⚠️  WARNING: Data is {days_old} days old. Run main.py to update!")
    
    # Fetch schedule
    schedule = fetch_schedule()
    print(f"   -> Found {len(schedule)} games with spreads")
    
    predictions = []
    skipped = []
    
    for g in schedule:
        # Match team names to historical data
        home_matched = find_best_match(g['home_raw'], known_teams)
        away_matched = find_best_match(g['away_raw'], known_teams)
        
        # Skip if we can't match teams or don't have stats
        if not home_matched or not away_matched:
            skipped.append(f"{g['away_raw']} @ {g['home_raw']} (Team matching failed)")
            continue
            
        if home_matched not in team_stats or away_matched not in team_stats:
            skipped.append(f"{g['away_raw']} @ {g['home_raw']} (No historical stats)")
            continue

        # Build feature row
        row = {'is_home': 1, 'spread': g['spread']}
        h_stats = team_stats[home_matched]
        a_stats = team_stats[away_matched]
        
        # Calculate rest days for BOTH teams
        home_last_date = pd.to_datetime(h_stats.get('last_game_date', datetime.now()))
        away_last_date = pd.to_datetime(a_stats.get('last_game_date', datetime.now()))
        
        home_actual_rest = max(0, (g['date'].replace(tzinfo=None) - home_last_date).days)
        away_actual_rest = max(0, (g['date'].replace(tzinfo=None) - away_last_date).days)
        
        # For model: use home team's rest, capped at 7 (if that's how it was trained)
        row['rest_days'] = min(home_actual_rest, 7)
        
        # Add production features
        row = calculate_production_features(row, h_stats, a_stats)
        
        # Prepare for model
        cols = model.feature_names_in_
        input_df = pd.DataFrame([row])
        for c in cols:
            if c not in input_df.columns: 
                input_df[c] = 0.0
        
        input_df.columns = input_df.columns.astype(str)
        
        # Make prediction
        prob = model.predict_proba(input_df)[0][1]
        conf = max(prob, 1-prob)
        
        # Determine pick - USE ORIGINAL ESPN NAMES
        if prob > 0.5:
            sign = "+" if g['spread'] > 0 else ""
            pick_str = f"{g['home_raw']} {sign}{g['spread']}"  # ← Original name
            picked_team_rest = home_actual_rest  # Picked home team
        else:
            away_spread = -1 * g['spread']
            sign = "+" if away_spread > 0 else ""
            pick_str = f"{g['away_raw']} {sign}{away_spread}"  # ← Original name
            picked_team_rest = away_actual_rest  # Picked away team

        # Format time in Eastern
        try:
            local_ts = g['date'].tz_convert('US/Eastern')
            time_str = local_ts.strftime("%m/%d %I:%M %p")
        except:
            time_str = g['date'].strftime("%m/%d %I:%M %p")

        # CRITICAL FIX: Use ORIGINAL ESPN names for display
        prediction_row = {
            "Date/Time": time_str,
            "Matchup": f"{g['away_raw']} @ {g['home_raw']}",  # ← ORIGINAL NAMES
            "Spread": g['spread'],
            "Pick": pick_str,
            "Conf": conf,
            "Raw Odds": g['raw_odds'],
            "Rest": picked_team_rest,  # ← Show PICKED TEAM's rest days
            # Debug fields (optional)
            "Home_Matched": home_matched,
            "Away_Matched": away_matched
        }
        
        # VALIDATION: Ensure pick mentions a team that's actually in the matchup
        pick_team_mentioned = pick_str.split()[0] + " " + pick_str.split()[1]
        if g['home_raw'] not in pick_str and g['away_raw'] not in pick_str:
            print(f"      ⚠️  WARNING: Pick '{pick_str}' doesn't match matchup '{prediction_row['Matchup']}'")
        
        predictions.append(prediction_row)

    # Save predictions
    if predictions:
        pred_df = pd.DataFrame(predictions).sort_values(by="Conf", ascending=False)
        
        # Save to current file (for app)
        pred_df.to_csv(OUTPUT_FILE, index=False)
        
        # ALSO save to dated archive file (for grading)
        archive_file = OUTPUT_FILE.replace("daily_predictions.csv", 
                                          f"predictions_{now_eastern.strftime('%Y%m%d')}.csv")
        pred_df.to_csv(archive_file, index=False)
        
        print(f"\n✅ SUCCESS: Generated {len(pred_df)} predictions")
        print(f"   Saved to: {OUTPUT_FILE}")
        print(f"   Archive: {archive_file}")
        
        # Show summary
        print("\n📋 PREDICTION SUMMARY:")
        for _, row in pred_df.head(5).iterrows():
            print(f"   {row['Matchup']}")
            print(f"      Pick: {row['Pick']} (Conf: {row['Conf']:.1%})")
    else:
        print("\n⚠️  No predictions generated.")
    
    # Show skipped games
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} games:")
        for s in skipped[:5]:
            print(f"   - {s}")

if __name__ == "__main__":
    main()