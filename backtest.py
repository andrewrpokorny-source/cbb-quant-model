import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.metrics import brier_score_loss

from league_config import get_league_artifact_paths, normalize_league
from model import FEATURES

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEEKS_BACK = 4

# High confidence threshold
HIGH_CONF_THRESHOLD = 0.53


def build_pipeline():
    """Build the production GBM + CalibratedClassifierCV pipeline."""
    base = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    return CalibratedClassifierCV(base, method='sigmoid', cv=5)


def train_model_at_date(df, cutoff_date, pipeline):
    """Train a model on all data before cutoff_date."""
    past_games = df[df['date'] < cutoff_date].copy()

    valid_feats = [f for f in FEATURES if f in past_games.columns]
    if len(valid_feats) != len(FEATURES):
        return None, None

    train_data = past_games.dropna(subset=valid_feats + ['ats_win'])

    if len(train_data) < 50:
        return None, None

    X = train_data[valid_feats].astype(float)
    y = train_data['ats_win'].astype(int)

    clf = clone(pipeline)
    clf.fit(X, y)

    return clf, valid_feats


def predict_week(model, feats, week_df):
    """Run predictions on a week of games. Returns DataFrame with results."""
    test = week_df.dropna(subset=feats).copy()
    if len(test) == 0:
        return None

    X_test = test[feats].astype(float)
    probs = model.predict_proba(X_test)[:, 1]

    test['prob_home'] = probs
    test['conf'] = test['prob_home'].apply(lambda x: max(x, 1 - x))

    conditions = [test['prob_home'] > 0.5, test['prob_home'] <= 0.5]
    test['picked_team'] = np.select(conditions, [test['team'], test['opponent']])
    test['picked_spread'] = np.select(conditions, [test['spread'], -1 * test['spread']])
    test['pick_correct'] = np.where(
        test['prob_home'] > 0.5,
        test['ats_win'] == 1,
        test['ats_win'] == 0,
    )

    return test[['date', 'picked_team', 'picked_spread', 'conf', 'pick_correct', 'prob_home', 'ats_win']]


def compute_metrics(log_df):
    """Compute all comparison metrics from a prediction log."""
    if log_df is None or len(log_df) == 0:
        return {
            'accuracy': 0.0,
            'brier': 1.0,
            'high_conf_acc': 0.0,
            'high_conf_bets': 0,
            'total_bets': 0,
            'roi_units': 0.0,
        }

    total = len(log_df)
    correct = log_df['pick_correct'].sum()
    accuracy = correct / total if total > 0 else 0.0

    brier = brier_score_loss(
        log_df['ats_win'].astype(int),
        log_df['prob_home'].astype(float),
    )

    hc = log_df[log_df['conf'] >= HIGH_CONF_THRESHOLD]
    hc_bets = len(hc)
    hc_acc = hc['pick_correct'].sum() / hc_bets if hc_bets > 0 else 0.0

    payout = 100.0 / 110.0
    hc_wins = hc['pick_correct'].sum()
    hc_losses = hc_bets - hc_wins
    roi_units = (hc_wins * payout) - (hc_losses * 1.0)

    return {
        'accuracy': accuracy,
        'brier': brier,
        'high_conf_acc': hc_acc,
        'high_conf_bets': hc_bets,
        'total_bets': total,
        'roi_units': roi_units,
    }


def run_backtest(league="mens"):
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    output_file = paths["performance_file"]

    print(f"--- WALK-FORWARD BACKTEST ({league}, GBM + Sigmoid, 15 features) ---")

    if not os.path.exists(data_file):
        print("CRITICAL ERROR: Training data not found at", data_file)
        return

    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Compute rest days dynamically
    df['last_game'] = df.groupby('team')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['last_game']).dt.days.fillna(7)
    df['rest_days'] = df['rest_days'].clip(upper=7)

    # Compute derived spread features
    df['spread_abs'] = df['spread'].abs()
    df['spread_squared'] = df['spread'] ** 2

    # Date range
    end_date = df['date'].max() + timedelta(days=1)
    start_date = end_date - timedelta(weeks=WEEKS_BACK)

    print(f"   Testing Range: {start_date.date()} to {end_date.date()}")
    print(f"   Weeks Back: {WEEKS_BACK}")
    print(f"   Config: depth=4, sigmoid, {len(FEATURES)} features")
    print()

    pipeline = build_pipeline()

    current_date = start_date
    logs = []

    while current_date < end_date:
        next_week = current_date + timedelta(days=7)
        week_label = current_date.date()

        model, valid_feats = train_model_at_date(df, current_date, pipeline)

        if model is None:
            print(f"   [SKIP] Week of {week_label}: not enough history to train.")
            current_date = next_week
            continue

        # Test on this week (home games only to avoid double-counting)
        mask = (df['date'] >= current_date) & (df['date'] < next_week) & (df['is_home'] == 1)
        week_df = df[mask].copy()

        if len(week_df) == 0:
            print(f"   [SKIP] Week of {week_label}: no games.")
            current_date = next_week
            continue

        result = predict_week(model, valid_feats, week_df)
        n = len(result) if result is not None else 0

        if result is not None:
            logs.append(result)

        print(f"   Week of {week_label}: {n} games")
        current_date = next_week

    full = pd.concat(logs) if logs else None

    if full is None:
        print("\nWARNING: Backtest ran but generated no bets.")
        return

    metrics = compute_metrics(full)

    # Print results
    print()
    print("=" * 55)
    print("=== BACKTEST RESULTS ===")
    print("=" * 55)
    print(f"{'Accuracy':<25} {metrics['accuracy']:.1%}")
    print(f"{'Brier Score':<25} {metrics['brier']:.4f}")
    print(f"{'High Conf Accuracy':<25} {metrics['high_conf_acc']:.1%}")
    print(f"{'High Conf Bets':<25} {metrics['high_conf_bets']}")
    print(f"{'Total Bets':<25} {metrics['total_bets']}")
    print(f"{'ROI (flat 1U @ -110)':<25} {metrics['roi_units']:+.2f}U")
    print("=" * 55)

    # Save performance log
    if full is not None:
        action_log = full[full['conf'] >= HIGH_CONF_THRESHOLD].copy()
        action_log[['date', 'picked_team', 'picked_spread', 'conf', 'pick_correct']].to_csv(
            output_file, index=False
        )
        print(f"\nSaved {len(action_log)} high-confidence bets to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run walk-forward backtest for CBB spread model.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League to backtest: mens or womens (aliases supported).",
    )
    args = parser.parse_args()
    run_backtest(args.league)
