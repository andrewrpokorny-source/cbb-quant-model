import pandas as pd
from difflib import get_close_matches

# --- CONFIG ---
STATS_FILE = "cbb_training_data_processed.csv"
ODDS_FILE = "espn_odds_history.csv"
OUTPUT_FILE = "cbb_training_data_with_totals.csv"

def normalize_name(name):
    """Simple normalizer to help matching."""
    return name.replace("St.", "State").replace("State", "St").lower()

def merge():
    print("--- 🔗 MERGING VEGAS ODDS WITH STATS ---")
    
    try:
        df_stats = pd.read_csv(STATS_FILE)
        df_odds = pd.read_csv(ODDS_FILE)
    except:
        print("❌ Missing input files. Run fetch_odds.py first."); return

    # Convert dates to match
    df_stats['date'] = pd.to_datetime(df_stats['date'])
    df_odds['date'] = pd.to_datetime(df_odds['date'])
    
    print(f"   Stats Rows: {len(df_stats)}")
    print(f"   Odds Rows:  {len(df_odds)}")

    # We need to match names. This is tricky.
    # We will iterate through the ODDS file and try to find the game in the STATS file.
    
    # Create a mapping dictionary for speed
    # Key: (Date, HomeTeamName) -> Value: Total
    odds_lookup = {}
    
    # Pre-process stats names for fuzzy matching
    stats_teams = df_stats['team'].unique()
    
    # Add a 'clean_total' column to stats, initialized to NaN
    df_stats['total_line'] = pd.NA
    
    matches_found = 0
    
    for idx, row in df_odds.iterrows():
        date = row['date']
        home_espn = row['home_team']
        total = row['total_line']
        
        # 1. Find the game in stats df for this date
        # Filter stats for this date and is_home=1
        daily_games = df_stats[(df_stats['date'] == date) & (df_stats['is_home'] == 1)]
        
        if daily_games.empty: continue
        
        # 2. Match Team Name
        # Try exact match first
        match = daily_games[daily_games['team'] == home_espn]
        
        if match.empty:
            # Try fuzzy match
            daily_teams = daily_games['team'].tolist()
            closest = get_close_matches(home_espn, daily_teams, n=1, cutoff=0.6)
            if closest:
                match = daily_games[daily_games['team'] == closest[0]]
        
        if not match.empty:
            # Update the Total Line in the main dataframe
            # We need to update BOTH Home and Away rows for that game
            team_name = match.iloc[0]['team']
            
            # Update Home
            mask_home = (df_stats['date'] == date) & (df_stats['team'] == team_name)
            df_stats.loc[mask_home, 'total_line'] = total
            
            # Update Away (Find row where opponent is this team on this date)
            mask_away = (df_stats['date'] == date) & (df_stats['opponent'] == team_name)
            df_stats.loc[mask_away, 'total_line'] = total
            
            matches_found += 1

    print(f"✅ Successfully merged lines for {matches_found} games.")
    
    # Recalculate 'total_over' target now that we have real lines
    df_stats['total_score'] = df_stats['team_score'] + df_stats['opp_score']
    df_stats['total_over'] = (df_stats['total_score'] > df_stats['total_line']).astype(int)
    
    # Save
    df_stats.to_csv(OUTPUT_FILE, index=False)
    print(f"   -> Saved to {OUTPUT_FILE}")
    print("   -> Now update your scripts to use this new file!")

if __name__ == "__main__":
    merge()