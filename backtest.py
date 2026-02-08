import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from datetime import timedelta
import os

from model import FEATURES

# --- BULLETPROOF PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "cbb_training_data_processed.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "performance_log.csv")
WEEKS_BACK = 4

# OLD config: 13 features (before spread_abs / spread_squared were added), depth=3, isotonic
OLD_FEATURES = [
    'is_home',
    'spread',
    'rest_days',
    'diff_eFG',
    'diff_Rebound',
    'diff_TO',
    'momentum_gap',
    'roll5_cover_margin',
    'prev_games_played',
    'opp_win_pct',
    'prev_blowout_rate',
    'prev_roll5_margin',
    'prev_volatility',
]

# NEW config: 15 features (full FEATURES list from model.py), depth=4, sigmoid
NEW_FEATURES = list(FEATURES)

# High confidence threshold
HIGH_CONF_THRESHOLD = 0.53


def build_pipeline(max_depth, cal_method, features):
    """Build a GradientBoosting + CalibratedClassifierCV pipeline matching production."""
    base = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=max_depth,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base, method=cal_method, cv=5)
    return calibrated, features


def train_model_at_date(df, cutoff_date, model_config):
    """Train a model on all data before cutoff_date using the given config."""
    calibrated_clf, features = model_config

    past_games = df[df['date'] < cutoff_date].copy()

    # Validate features exist
    valid_feats = [f for f in features if f in past_games.columns]
    if len(valid_feats) != len(features):
        return None, None

    train_data = past_games.dropna(subset=valid_feats + ['ats_win'])

    if len(train_data) < 50:
        return None, None

    X = train_data[valid_feats].astype(float)
    y = train_data['ats_win'].astype(int)

    # Clone the pipeline so we get a fresh unfitted estimator each week
    from sklearn.base import clone
    clf = clone(calibrated_clf)
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

    # Pick logic: if prob_home > 0.5, pick home; else pick away
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

    # Brier score: use prob_home vs ats_win
    brier = brier_score_loss(
        log_df['ats_win'].astype(int),
        log_df['prob_home'].astype(float),
    )

    # High confidence subset
    hc = log_df[log_df['conf'] >= HIGH_CONF_THRESHOLD]
    hc_bets = len(hc)
    hc_acc = hc['pick_correct'].sum() / hc_bets if hc_bets > 0 else 0.0

    # ROI simulation: flat 1U bets at -110 odds
    # Win pays +1/1.10 = +0.909..., Loss costs -1U
    payout = 100.0 / 110.0  # ~0.9091
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


def run_backtest():
    print("--- STARTING BACKTEST: OLD vs NEW MODEL COMPARISON ---")

    if not os.path.exists(DATA_FILE):
        print("CRITICAL ERROR: Training data not found at", DATA_FILE)
        return

    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Compute rest days dynamically
    df['last_game'] = df.groupby('team')['date'].shift(1)
    df['rest_days'] = (df['date'] - df['last_game']).dt.days.fillna(7)
    df['rest_days'] = df['rest_days'].clip(upper=7)

    # Compute derived spread features needed by the new model
    df['spread_abs'] = df['spread'].abs()
    df['spread_squared'] = df['spread'] ** 2

    # Date range
    end_date = df['date'].max() + timedelta(days=1)
    start_date = end_date - timedelta(weeks=WEEKS_BACK)

    print(f"   Testing Range: {start_date.date()} to {end_date.date()}")
    print(f"   Weeks Back: {WEEKS_BACK}")
    print(f"   Old Config: depth=3, isotonic, {len(OLD_FEATURES)} features")
    print(f"   New Config: depth=4, sigmoid, {len(NEW_FEATURES)} features")
    print()

    # Build both model configs
    old_pipeline, old_feats = build_pipeline(max_depth=3, cal_method='isotonic', features=OLD_FEATURES)
    new_pipeline, new_feats = build_pipeline(max_depth=4, cal_method='sigmoid', features=NEW_FEATURES)

    current_date = start_date
    old_logs = []
    new_logs = []

    while current_date < end_date:
        next_week = current_date + timedelta(days=7)
        week_label = current_date.date()

        # Train both models on all data before current_date
        old_model, old_valid_feats = train_model_at_date(df, current_date, (old_pipeline, old_feats))
        new_model, new_valid_feats = train_model_at_date(df, current_date, (new_pipeline, new_feats))

        if old_model is None or new_model is None:
            label = "old" if old_model is None else "new"
            if old_model is None and new_model is None:
                label = "both"
            print(f"   [SKIP] Week of {week_label}: not enough history to train ({label}).")
            current_date = next_week
            continue

        # Test on this week (home games only to avoid double-counting)
        mask = (df['date'] >= current_date) & (df['date'] < next_week) & (df['is_home'] == 1)
        week_df = df[mask].copy()

        if len(week_df) == 0:
            print(f"   [SKIP] Week of {week_label}: no games.")
            current_date = next_week
            continue

        # Run predictions with both models
        old_result = predict_week(old_model, old_valid_feats, week_df)
        new_result = predict_week(new_model, new_valid_feats, week_df)

        old_n = len(old_result) if old_result is not None else 0
        new_n = len(new_result) if new_result is not None else 0

        if old_result is not None:
            old_logs.append(old_result)
        if new_result is not None:
            new_logs.append(new_result)

        print(f"   Week of {week_label}: old={old_n} games, new={new_n} games")

        current_date = next_week

    # Combine results
    old_full = pd.concat(old_logs) if old_logs else None
    new_full = pd.concat(new_logs) if new_logs else None

    if old_full is None and new_full is None:
        print("\nWARNING: Backtest ran but generated no bets for either model.")
        return

    # Compute metrics
    old_metrics = compute_metrics(old_full)
    new_metrics = compute_metrics(new_full)

    # Print comparison table
    print()
    print("=" * 75)
    print("=== OLD vs NEW MODEL COMPARISON ===")
    print("=" * 75)
    print(f"{'Metric':<25} {'Old (d=3,iso,13f)':<22} {'New (d=4,sig,15f)':<22} {'Change':<12}")
    print("-" * 75)

    # Accuracy
    old_acc = old_metrics['accuracy']
    new_acc = new_metrics['accuracy']
    print(f"{'Accuracy':<25} {old_acc:<22.1%} {new_acc:<22.1%} {(new_acc - old_acc) * 100:+.1f}%")

    # Brier Score (lower is better)
    old_brier = old_metrics['brier']
    new_brier = new_metrics['brier']
    print(f"{'Brier Score':<25} {old_brier:<22.4f} {new_brier:<22.4f} {new_brier - old_brier:+.4f}")

    # High Conf Accuracy
    old_hc_acc = old_metrics['high_conf_acc']
    new_hc_acc = new_metrics['high_conf_acc']
    print(f"{'High Conf Accuracy':<25} {old_hc_acc:<22.1%} {new_hc_acc:<22.1%} {(new_hc_acc - old_hc_acc) * 100:+.1f}%")

    # High Conf Bets
    old_hc_bets = old_metrics['high_conf_bets']
    new_hc_bets = new_metrics['high_conf_bets']
    print(f"{'High Conf Bets':<25} {old_hc_bets:<22} {new_hc_bets:<22} {new_hc_bets - old_hc_bets:+d}")

    # Total Bets
    old_total = old_metrics['total_bets']
    new_total = new_metrics['total_bets']
    print(f"{'Total Bets':<25} {old_total:<22} {new_total:<22} {new_total - old_total:+d}")

    # ROI
    old_roi = old_metrics['roi_units']
    new_roi = new_metrics['roi_units']
    print(f"{'ROI (flat 1U @ -110)':<25} {old_roi:<+22.2f}U {new_roi:<+21.2f}U {new_roi - old_roi:+.2f}U")

    print("=" * 75)

    # Save NEW model performance log (for continuity with existing workflow)
    if new_full is not None:
        action_log = new_full[new_full['conf'] >= HIGH_CONF_THRESHOLD].copy()
        action_log[['date', 'picked_team', 'picked_spread', 'conf', 'pick_correct']].to_csv(
            OUTPUT_FILE, index=False
        )
        print(f"\nSaved {len(action_log)} high-confidence bets (new model) to {OUTPUT_FILE}")
    else:
        print("\nNo new model bets to save.")


if __name__ == "__main__":
    run_backtest()
