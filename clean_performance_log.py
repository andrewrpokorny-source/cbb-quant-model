#!/usr/bin/env python3
"""
Clean Performance Log - Remove Low Confidence Bets

This script removes all bets with confidence < 53% from the performance log.
Run this after updating to the confidence-filtered grading system.
"""

import pandas as pd
import os
from datetime import datetime

PERF_FILE = "performance_log.csv"
CONFIDENCE_THRESHOLD = 0.53

print("="*60)
print("CLEANING PERFORMANCE LOG")
print("="*60)

if not os.path.exists(PERF_FILE):
    print(f"\n❌ {PERF_FILE} not found")
    print("   Nothing to clean. Run grade_predictions.py first.")
    exit(0)

# Load current performance log
df = pd.read_csv(PERF_FILE)
original_count = len(df)

print(f"\nOriginal performance log:")
print(f"   Total bets: {original_count}")

# Filter for actionable bets only
df_filtered = df[df['conf'] >= CONFIDENCE_THRESHOLD].copy()
filtered_count = len(df_filtered)
removed_count = original_count - filtered_count

print(f"\nAfter filtering (conf >= {CONFIDENCE_THRESHOLD:.0%}):")
print(f"   Remaining bets: {filtered_count}")
print(f"   Removed: {removed_count}")

if removed_count > 0:
    # Backup original
    backup_file = f"performance_log_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False)
    print(f"\n💾 Backed up original to: {backup_file}")
    
    # Save filtered version
    df_filtered.to_csv(PERF_FILE, index=False)
    print(f"✅ Saved filtered log to: {PERF_FILE}")
    
    # Show updated stats
    if len(df_filtered) > 0:
        wins = df_filtered['pick_correct'].sum()
        win_rate = wins / len(df_filtered)
        units = df_filtered.apply(lambda x: 1.0 if x['pick_correct'] else -1.1, axis=1).sum()
        
        print(f"\n📊 Updated Performance (Actionable Bets Only):")
        print(f"   Record: {int(wins)}-{len(df_filtered)-int(wins)}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Profit: {units:+.2f} units")
else:
    print("\n✅ No low-confidence bets found. Log is already clean.")

print("\n" + "="*60)
print("DONE")
print("="*60)
print("\nNext steps:")
print("  1. Refresh your Streamlit dashboard")
print("  2. Performance metrics will now show only actionable bets")