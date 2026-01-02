import pandas as pd

# Load the PROCESSED data (the one causing issues)
df = pd.read_csv("cbb_training_data_processed.csv")
print(f"Total Rows: {len(df)}")

# Check for NaNs in key columns
columns_to_check = [
    'season_team_eFG', 
    'opp_season_team_eFG', # <--- If this is high, the Merge is failing
    'roll5_cover_margin',
    'diff_eFG'
]

print("\n--- 🔍 MISSING DATA REPORT ---")
for col in columns_to_check:
    if col in df.columns:
        missing = df[col].isna().sum()
        print(f"{col}: Missing {missing} rows ({missing/len(df):.1%})")
    else:
        print(f"{col}: NOT FOUND")