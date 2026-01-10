import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import pytz
from difflib import get_close_matches

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_FILE = os.path.join(BASE_DIR, "daily_predictions.csv")
PERF_FILE = os.path.join(BASE_DIR, "performance_log.csv")
BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?groups=50&limit=1000"

def normalize_team_name(name):
    """Normalize team names for matching."""
    # Remove common suffixes
    name = name.replace(" Tigers", "").replace(" Bulldogs", "").replace(" Eagles", "")
    name = name.replace(" Wildcats", "").replace(" Cardinals", "")
    # Remove state suffixes
    name = name.replace(" State", "").replace(" St.", "").replace(" St", "")
    return name.strip()

def fetch_completed_games(date_obj):
    """
    Fetch completed games for a specific date from ESPN.
    Returns dict: {(home_team, away_team): {home_score, away_score, spread}}
    """
    print(f"   -> Fetching completed games for {date_obj.strftime('%Y-%m-%d')}...")
    
    date_str = date_obj.strftime("%Y%m%d")
    url = f"{BASE_URL}&dates={date_str}"
    
    games = {}
    
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
        
        for event in data.get('events', []):
            # Only process completed games
            status = event['status']['type']['state']
            if status != 'post':
                continue
            
            comp = event['competitions'][0]
            if not comp.get('competitors'):
                continue
            
            home_tm = comp['competitors'][0]
            away_tm = comp['competitors'][1]
            
            home_name = home_tm['team']['displayName']
            away_name = away_tm['team']['displayName']
            
            home_score = int(home_tm['score'])
            away_score = int(away_tm['score'])
            
            # Get the spread (if available)
            spread = 0.0
            if comp.get('odds'):
                odds = comp['odds'][0]
                details = odds.get('details', '0')
                try:
                    if details and details != '0' and details != 'EVEN':
                        parts = details.split()
                        val = abs(float(parts[-1]))
                        fav = " ".join(parts[:-1])
                        
                        home_abbr = home_tm['team'].get('abbreviation', '')
                        is_home_fav = (fav == home_abbr) or (fav == home_name) or (fav in home_name)
                        
                        spread = -val if is_home_fav else val
                except:
                    pass
            
            # Store with multiple key formats for easier matching
            game_key = (home_name, away_name)
            games[game_key] = {
                'home_score': home_score,
                'away_score': away_score,
                'spread': spread,
                'home_name': home_name,
                'away_name': away_name
            }
            
        print(f"      Found {len(games)} completed games")
        return games
        
    except Exception as e:
        print(f"      ❌ Error fetching games: {e}")
        return {}

def match_prediction_to_game(pred_matchup, games):
    """
    Try to match a prediction matchup to an actual game.
    pred_matchup format: "Away @ Home"
    """
    # Parse prediction matchup
    parts = pred_matchup.split(' @ ')
    if len(parts) != 2:
        return None
    
    pred_away, pred_home = parts
    
    # Try exact match first
    for (game_home, game_away), result in games.items():
        if game_home == pred_home and game_away == pred_away:
            return result
    
    # Try fuzzy matching
    for (game_home, game_away), result in games.items():
        # Check if key parts of names match
        if (pred_home in game_home or game_home in pred_home) and \
           (pred_away in game_away or game_away in pred_away):
            return result
    
    return None

def grade_pick(pick_str, spread, home_score, away_score, matchup):
    """
    Determine if a pick was correct.
    pick_str format: "Team Name +/-X.X"
    """
    parts = matchup.split(' @ ')
    if len(parts) != 2:
        return None
    
    away_team, home_team = parts
    
    # Determine which team was picked
    if home_team in pick_str:
        # Picked home team
        picked_home = True
        ats_margin = home_score + spread - away_score
    else:
        # Picked away team
        picked_home = False
        ats_margin = away_score - spread - home_score
    
    # Did the pick win?
    pick_won = ats_margin > 0
    
    return pick_won

def grade_predictions():
    """
    Main function to grade yesterday's predictions.
    """
    print("="*60)
    print("GRADING YESTERDAY'S PREDICTIONS")
    print("="*60)
    
    # 1. Determine yesterday's date (Eastern time)
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern)
    yesterday = today - timedelta(days=1)
    yesterday_date = yesterday.date()
    
    print(f"\nToday: {today.strftime('%Y-%m-%d %I:%M %p %Z')}")
    print(f"Grading date: {yesterday_date}\n")
    
    # 2. Check if we have predictions file
    if not os.path.exists(PRED_FILE):
        print("❌ No predictions file found (daily_predictions.csv)")
        print("   Run predict.py to generate predictions first.")
        return
    
    # 3. Load predictions
    try:
        preds = pd.read_csv(PRED_FILE)
        print(f"✅ Loaded {len(preds)} predictions from file")
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return
    
    # 4. Parse prediction dates
    # The Date/Time column might be "01/09 07:00 PM" format
    # We need to figure out which predictions are for yesterday
    
    # For now, let's assume ALL predictions in the file are recent
    # (since predict.py overwrites daily_predictions.csv each time)
    # We'll match them to yesterday's completed games
    
    print(f"\n📊 Predictions to grade: {len(preds)}")
    
    # 5. Fetch yesterday's completed games
    completed_games = fetch_completed_games(yesterday)
    
    if not completed_games:
        print("\n❌ No completed games found for yesterday.")
        print("   Either no games were played, or ESPN API is not responding.")
        return
    
    # 6. Grade each prediction
    print(f"\n📝 Grading predictions...")
    
    graded_bets = []
    unmatched = []
    
    for idx, pred in preds.iterrows():
        matchup = pred['Matchup']
        pick = pred['Pick']
        conf = pred['Conf']
        spread = pred.get('Spread', 0.0)
        
        # Find the corresponding game
        game_result = match_prediction_to_game(matchup, completed_games)
        
        if game_result is None:
            # Couldn't find this game - might be for today/tomorrow
            unmatched.append(matchup)
            continue
        
        # Grade the pick
        pick_correct = grade_pick(
            pick, 
            game_result['spread'],
            game_result['home_score'],
            game_result['away_score'],
            matchup
        )
        
        if pick_correct is None:
            unmatched.append(matchup)
            continue
        
        # Extract picked team and spread
        pick_parts = pick.split()
        picked_team = " ".join(pick_parts[:-1])
        picked_spread = float(pick_parts[-1])
        
        graded_bets.append({
            'date': yesterday_date,
            'picked_team': picked_team,
            'picked_spread': picked_spread,
            'conf': conf,
            'pick_correct': pick_correct,
            'matchup': matchup,
            'final_score': f"{game_result['away_score']}-{game_result['home_score']}"
        })
        
        result_icon = "✅" if pick_correct else "❌"
        print(f"   {result_icon} {matchup}: {pick} ({'WIN' if pick_correct else 'LOSS'})")
    
    print(f"\n📊 Grading Summary:")
    print(f"   Graded: {len(graded_bets)}")
    print(f"   Unmatched: {len(unmatched)}")
    
    if unmatched:
        print(f"\n   Unmatched predictions (likely for today/tomorrow):")
        for m in unmatched[:5]:
            print(f"      - {m}")
    
    # 7. Save to performance log
    if graded_bets:
        graded_df = pd.DataFrame(graded_bets)
        
        # Append to existing performance log or create new one
        if os.path.exists(PERF_FILE):
            existing = pd.read_csv(PERF_FILE)
            existing['date'] = pd.to_datetime(existing['date']).dt.date
            
            # Remove any existing entries for yesterday (in case re-grading)
            existing = existing[existing['date'] != yesterday_date]
            
            # Combine
            combined = pd.concat([existing, graded_df], ignore_index=True)
            combined.to_csv(PERF_FILE, index=False)
            print(f"\n✅ Added {len(graded_bets)} graded bets to performance_log.csv")
        else:
            graded_df.to_csv(PERF_FILE, index=False)
            print(f"\n✅ Created performance_log.csv with {len(graded_bets)} bets")
        
        # Show win rate
        wins = sum(graded_df['pick_correct'])
        win_rate = wins / len(graded_df)
        profit = sum(1.0 if x else -1.1 for x in graded_df['pick_correct'])
        
        print(f"\n🎯 Yesterday's Performance:")
        print(f"   Record: {wins}-{len(graded_df)-wins}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Profit: {profit:+.2f} units")
    else:
        print("\n⚠️  No predictions were graded.")
        print("   This likely means the predictions file contains games for today/tomorrow.")

if __name__ == "__main__":
    grade_predictions()