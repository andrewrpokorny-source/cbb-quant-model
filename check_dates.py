import pandas as pd
import os

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")

def main():
    print("--- 🕵️ DATA DIAGNOSTIC ---")
    
    if not os.path.exists(DATA_FILE):
        print("❌ No data file found.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"📊 Total Rows: {len(df)}")
    
    # 1. Check Date Type
    print(f"   -> Date Column Type: {df['date'].dtype}")
    
    # 2. Show the Last 10 Dates (Raw)
    print("\n   -> Last 10 Dates in File (Raw):")
    print(df['date'].tail(10).to_string(index=False))
    
    # 3. Check for Jan 7 specifically
    # We look for partial matches to see if it's there but weird
    mask = df['date'].astype(str).str.contains("2026-01-07")
    jan7 = df[mask]
    
    print(f"\n   -> Rows containing '2026-01-07': {len(jan7)}")
    
    if len(jan7) > 0:
        print("   -> Sample Jan 7 Row:")
        print(jan7.iloc[0][['date', 'team', 'opponent']])
    else:
        print("   -> ⚠️ ZERO rows found for Jan 7. The day is missing.")

if __name__ == "__main__":
    main()