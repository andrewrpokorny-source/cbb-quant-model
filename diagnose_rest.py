import pandas as pd
from datetime import datetime, timedelta
import pytz

print("="*60)
print("REST DAYS DIAGNOSTIC")
print("="*60)

# 1. Check historical data
print("\n1. CHECKING HISTORICAL DATA:")
try:
    df = pd.read_csv('cbb_training_data_processed.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    last_date = df['date'].max()
    print(f"   Data current through: {last_date.strftime('%Y-%m-%d')}")
    
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern).date()
    days_behind = (today - last_date.date()).days
    
    if days_behind > 1:
        print(f"   ⚠️  WARNING: Data is {days_behind} days behind!")
        print(f"   Action needed: Run 'python3 main.py' to update")
    else:
        print(f"   ✅ Data is current (only {days_behind} days behind)")
    
except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    exit(1)

# 2. Check a few specific teams
print("\n2. CHECKING SPECIFIC TEAMS (Last 3 Games):")
test_teams = ['Duke', 'Houston', 'Kansas', 'North Carolina', 'Auburn']

for team in test_teams:
    team_games = df[df['team'] == team].sort_values('date', ascending=False).head(3)
    if len(team_games) > 0:
        print(f"\n   {team}:")
        for _, game in team_games.iterrows():
            print(f"      {game['date'].strftime('%Y-%m-%d')}: vs {game['opponent']}")
        
        last_game = team_games.iloc[0]['date']
        days_since = (today - last_game.date()).days
        print(f"      → Last game was {days_since} days ago")
    else:
        print(f"\n   {team}: NOT FOUND in data")

# 3. Check predictions file rest days
print("\n3. CHECKING CURRENT PREDICTIONS:")
try:
    pred = pd.read_csv('daily_predictions.csv')
    
    print(f"   Total predictions: {len(pred)}")
    print(f"\n   Rest Days Distribution:")
    rest_counts = pred['Rest'].value_counts().sort_index()
    for rest, count in rest_counts.items():
        pct = (count / len(pred)) * 100
        bar = "█" * int(pct / 2)
        print(f"      {int(rest)} days: {count:3d} games ({pct:5.1f}%) {bar}")
    
    # Show examples
    print(f"\n   Sample Predictions:")
    for _, row in pred.head(10).iterrows():
        print(f"      {row['Matchup'][:40]:40s} Rest: {row['Rest']}")
        
except Exception as e:
    print(f"   ❌ Error loading predictions: {e}")

# 4. Manual calculation test
print("\n4. MANUAL REST CALCULATION TEST:")
print("   Testing Houston (if in data):")

houston_games = df[df['team'] == 'Houston'].sort_values('date', ascending=False)
if len(houston_games) > 0:
    last_houston_game = houston_games.iloc[0]['date']
    print(f"      Last game: {last_houston_game.strftime('%Y-%m-%d')}")
    
    # Simulate today's game
    today_dt = pd.Timestamp.now(tz='US/Eastern').normalize()
    actual_rest = (today_dt - last_houston_game).days
    print(f"      Today's date: {today_dt.strftime('%Y-%m-%d')}")
    print(f"      Calculated rest: {actual_rest} days")
    
    if actual_rest >= 7:
        print(f"      ⚠️  REST IS ACTUALLY 7+ DAYS!")
        print(f"      This means Houston hasn't played since {last_houston_game.strftime('%Y-%m-%d')}")
        print(f"      Check if main.py successfully downloaded recent games")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)

print("\n💡 RECOMMENDATIONS:")
print("   1. If 'Data current through' is old → Run: python3 main.py")
print("   2. If teams show 7+ days rest → They genuinely haven't played")
print("   3. If you updated main.py recently → Check for API errors")