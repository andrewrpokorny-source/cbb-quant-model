import argparse
import os
import sys
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from scipy.stats import norm

from league_config import get_league_artifact_paths, normalize_league

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALIDATION_WEEKS_BACK = 4
HIGH_CONF_THRESHOLD = 0.53

if __name__ == "__main__":
    sys.modules.setdefault("model", sys.modules[__name__])

FEATURES = [
    'is_home',
    'is_neutral',
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
    'spread_abs',
    'spread_squared',
]
MENS_FEATURES = [
    'is_home',
    'is_neutral',
    'spread',
    'rest_days',
    'torvik_diff_adj_oe',
    'torvik_diff_adj_de',
    'torvik_diff_barthag',
    'torvik_tempo_gap',
    'torvik_diff_efg',
    'torvik_diff_tor',
    'torvik_diff_orb',
    'torvik_diff_ftr',
    'roll5_cover_margin',
    'prev_games_played',
    'opp_win_pct',
    'prev_blowout_rate',
    'prev_roll5_margin',
    'prev_volatility',
    'spread_abs',
    'spread_squared',
]
FEATURES_BY_LEAGUE = {
    'mens': MENS_FEATURES,
    'womens': FEATURES,
}
CALIBRATION_BY_LEAGUE = {
    'mens': False,
    'womens': True,
}


class TimeAwareCalibratedGBM(BaseEstimator, ClassifierMixin):
    """GBM with trailing-window sigmoid calibration instead of random CV folds."""

    def __init__(
        self,
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        calibration_fraction=0.2,
        min_calibration_rows=200,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.calibration_fraction = calibration_fraction
        self.min_calibration_rows = min_calibration_rows

    def _base_estimator(self):
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy()
        y_ser = pd.Series(y).astype(int).reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)
        self.feature_names_in_ = X_df.columns.astype(str).to_numpy()
        self.n_features_in_ = len(self.feature_names_in_)

        split_idx = max(1, int(len(X_df) * (1 - self.calibration_fraction)))
        split_idx = min(split_idx, len(X_df) - 1)

        use_calibration = (
            len(X_df) >= self.min_calibration_rows
            and split_idx < len(X_df)
            and y_ser.iloc[:split_idx].nunique() > 1
            and y_ser.iloc[split_idx:].nunique() > 1
        )

        base_X = X_df.iloc[:split_idx] if use_calibration else X_df
        base_y = y_ser.iloc[:split_idx] if use_calibration else y_ser

        self.base_estimator_ = self._base_estimator()
        self.base_estimator_.fit(base_X, base_y)
        self.classes_ = np.array([0, 1])

        self.calibrator_ = None
        self.calibration_rows_ = 0
        if use_calibration:
            calib_X = X_df.iloc[split_idx:]
            calib_y = y_ser.iloc[split_idx:]
            raw = np.clip(self.base_estimator_.predict_proba(calib_X)[:, 1], 1e-6, 1 - 1e-6)
            self.calibrator_ = LogisticRegression(solver="lbfgs")
            self.calibrator_.fit(raw.reshape(-1, 1), calib_y)
            self.calibration_rows_ = len(calib_X)

        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X).copy()
        raw = np.clip(self.base_estimator_.predict_proba(X_df)[:, 1], 1e-6, 1 - 1e-6)
        if self.calibrator_ is not None:
            calibrated = self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            calibrated = raw
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.base_estimator_.feature_importances_


# Keep pickle module path stable when model.py is executed as a script.
TimeAwareCalibratedGBM.__module__ = "model"


def _build_game_keys(df_model):
    return (
        df_model["_model_date"].dt.strftime("%Y-%m-%d")
        + "::"
        + df_model[["team", "opponent"]].fillna("").apply(
            lambda row: "||".join(sorted(row.tolist())),
            axis=1,
        )
    )


def _home_row_mask(df_model):
    if "is_home" in df_model.columns:
        return pd.to_numeric(df_model["is_home"], errors="coerce").fillna(0).astype(int) == 1
    if "location" in df_model.columns:
        return df_model["location"].astype(str).str.lower() == "home"
    raise ValueError("Training data must include either 'is_home' or 'location'.")


def prepare_time_ordered_training_frame(df, features, target):
    """Return a clean model frame sorted in chronological order.

    This enforces date ordering before any train/test split so the holdout set
    always contains the latest games rather than relying on CSV row order.
    """
    df_model = df.dropna(subset=features + [target]).copy()
    if 'date' not in df_model.columns:
        raise ValueError("Training data must include a 'date' column for time-aware splits.")

    df_model['_model_date'] = pd.to_datetime(df_model['date'], errors='coerce')
    df_model = df_model.dropna(subset=['_model_date']).copy()
    df_model['_original_order'] = np.arange(len(df_model))
    df_model = df_model.sort_values(['_model_date', '_original_order']).reset_index(drop=True)
    if "team" not in df_model.columns or "opponent" not in df_model.columns:
        raise ValueError("Training data must include 'team' and 'opponent' columns for game-level splits.")
    df_model["_game_key"] = _build_game_keys(df_model)
    return df_model


def time_series_train_test_split(df_model, features, target, test_size=0.2, bet_level_test=False):
    """Split a time-ordered frame into chronological game-level train/test partitions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(df_model) < 2:
        raise ValueError("Need at least two rows for a chronological train/test split.")

    game_dates = (
        df_model.groupby("_game_key")["_model_date"]
        .min()
        .sort_values()
    )
    if len(game_dates) < 2:
        raise ValueError("Need at least two games for a chronological train/test split.")

    test_games = max(1, int(len(game_dates) * test_size))
    test_keys = set(game_dates.index[-test_games:])
    train_mask = ~df_model["_game_key"].isin(test_keys)
    test_mask = df_model["_game_key"].isin(test_keys)
    if bet_level_test:
        test_mask = test_mask & _home_row_mask(df_model)

    X = df_model[features].astype(float)
    y = df_model[target].astype(int)
    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()
    return X_train, X_test, y_train, y_test


def _prepare_training_data(df, features, target):
    frame = df.dropna(subset=features + [target]).copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame


def _compute_validation_metrics(log_df):
    if log_df is None or len(log_df) == 0:
        return {
            "accuracy": 0.0,
            "brier": 1.0,
            "high_conf_acc": 0.0,
            "high_conf_bets": 0,
            "roi_units": 0.0,
            "total_bets": 0,
        }

    accuracy = float(log_df["pick_correct"].mean())
    brier = float(brier_score_loss(log_df["ats_win"].astype(int), log_df["prob_home"].astype(float)))
    high_conf = log_df[log_df["conf"] >= HIGH_CONF_THRESHOLD]
    high_conf_bets = int(len(high_conf))
    high_conf_acc = float(high_conf["pick_correct"].mean()) if high_conf_bets else 0.0
    payout = 100.0 / 110.0
    wins = float(high_conf["pick_correct"].sum())
    losses = high_conf_bets - wins
    roi_units = (wins * payout) - losses
    return {
        "accuracy": accuracy,
        "brier": brier,
        "high_conf_acc": high_conf_acc,
        "high_conf_bets": high_conf_bets,
        "roi_units": roi_units,
        "total_bets": int(len(log_df)),
    }


def walk_forward_validate(df, features, target, estimator_factory, weeks_back=VALIDATION_WEEKS_BACK):
    df_eval = _prepare_training_data(df, features, target)
    if df_eval.empty:
        return _compute_validation_metrics(None)

    end_date = df_eval["date"].max() + timedelta(days=1)
    start_date = end_date - timedelta(weeks=weeks_back)
    current_date = start_date
    logs = []

    while current_date < end_date:
        next_week = current_date + timedelta(days=7)
        train_data = df_eval[df_eval["date"] < current_date].copy()
        if len(train_data) < 50:
            current_date = next_week
            continue

        week_mask = (
            (df_eval["date"] >= current_date)
            & (df_eval["date"] < next_week)
            & (pd.to_numeric(df_eval.get("is_home"), errors="coerce").fillna(0).astype(int) == 1)
        )
        week_df = df_eval.loc[week_mask].copy()
        if week_df.empty:
            current_date = next_week
            continue

        estimator = estimator_factory()
        estimator.fit(train_data[features].astype(float), train_data[target].astype(int))
        probs = estimator.predict_proba(week_df[features].astype(float))[:, 1]
        conf = np.maximum(probs, 1 - probs)
        pick_correct = np.where(probs > 0.5, week_df[target].astype(int) == 1, week_df[target].astype(int) == 0)
        logs.append(
            pd.DataFrame(
                {
                    "date": week_df["date"].values,
                    "prob_home": probs,
                    "conf": conf,
                    "pick_correct": pick_correct,
                    "ats_win": week_df[target].astype(int).values,
                }
            )
        )
        current_date = next_week

    full_log = pd.concat(logs, ignore_index=True) if logs else None
    return _compute_validation_metrics(full_log)


def _print_walk_forward_comparison(calibrated_metrics, uncalibrated_metrics):
    print(f"\n=== WALK-FORWARD VALIDATION ({VALIDATION_WEEKS_BACK} WEEKS) ===")
    print(f"{'Metric':<22} {'Uncalibrated':<15} {'Calibrated':<15} {'Preferred':<12}")
    print(f"{'-'*70}")

    comparisons = [
        ("Accuracy", f"{uncalibrated_metrics['accuracy']:.1%}", f"{calibrated_metrics['accuracy']:.1%}", "higher"),
        ("Brier Score", f"{uncalibrated_metrics['brier']:.4f}", f"{calibrated_metrics['brier']:.4f}", "lower"),
        (
            f"High Conf (>{HIGH_CONF_THRESHOLD:.0%})",
            f"{uncalibrated_metrics['high_conf_acc']:.1%} ({uncalibrated_metrics['high_conf_bets']})",
            f"{calibrated_metrics['high_conf_acc']:.1%} ({calibrated_metrics['high_conf_bets']})",
            "higher",
        ),
        (
            "ROI Units",
            f"{uncalibrated_metrics['roi_units']:+.2f}U",
            f"{calibrated_metrics['roi_units']:+.2f}U",
            "higher",
        ),
    ]

    for label, uncal_value, cal_value, preferred in comparisons:
        print(f"{label:<22} {uncal_value:<15} {cal_value:<15} {preferred:<12}")


def use_calibrated_spread_model(league="mens"):
    """Return whether the production spread model should use trailing calibration."""
    return bool(CALIBRATION_BY_LEAGUE[normalize_league(league)])


def build_spread_estimator(league="mens", calibrated=None):
    """Build the league-specific production spread estimator."""
    league = normalize_league(league)
    if calibrated is None:
        calibrated = use_calibrated_spread_model(league)

    if calibrated:
        return TimeAwareCalibratedGBM(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )

    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )

def train_and_evaluate(league="mens"):
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    model_file = paths["model_file"]
    features = get_feature_list(league)
    production_calibrated = use_calibrated_spread_model(league)
    production_label = "GBM + Sigmoid Calibration" if production_calibrated else "GBM"

    print(f"--- TRAINING CBB MODEL ({league}, {production_label}, {len(features)} features) ---")

    if not os.path.exists(data_file):
        print("No processed data found. Run features.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(data_file, low_memory=False)
    print(f"Loaded {len(df)} rows.")

    # Compute derived spread features
    df['spread_abs'] = df['spread'].abs()
    df['spread_squared'] = df['spread'] ** 2

    # 2. Define Features
    target = 'ats_win'

    # 3. Validation: Ensure all columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"CRITICAL ERROR: Missing features in CSV: {missing_cols}")
        print("Run 'python3 features.py' again to regenerate them.")
        return

    # 4. Clean, validate, and sort chronologically before splitting
    df_model = prepare_time_ordered_training_frame(df, features, target)

    print(f"Training on {len(df_model)} clean games.")
    print(f"Features: {len(features)}")

    # 5. Split data: train / test (time-series aware diagnostic holdout)
    X_train, X_test, y_train, y_test = time_series_train_test_split(
        df_model, features, target, test_size=0.2, bet_level_test=True
    )

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 6. Train base model (uncalibrated) for comparison
    base_clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    base_clf.fit(X_train, y_train)

    # 7. Evaluate UNCALIBRATED model
    uncal_probs = base_clf.predict_proba(X_test)[:, 1]
    uncal_preds = base_clf.predict(X_test)
    uncal_acc = accuracy_score(y_test, uncal_preds)
    uncal_brier = brier_score_loss(y_test, uncal_probs)

    # 8. Train CALIBRATED model using cross-validation
    # CalibratedClassifierCV with cv=5 will handle calibration internally
    calibrated_clf = TimeAwareCalibratedGBM(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    calibrated_clf.fit(X_train, y_train)

    # 9. Evaluate CALIBRATED model
    cal_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    cal_preds = calibrated_clf.predict(X_test)
    cal_acc = accuracy_score(y_test, cal_preds)
    cal_brier = brier_score_loss(y_test, cal_probs)

    # 10. Production-aligned walk-forward validation on the trailing weeks
    calibrated_metrics = walk_forward_validate(
        df,
        features,
        target,
        estimator_factory=lambda: build_spread_estimator(league, calibrated=True),
    )
    uncalibrated_metrics = walk_forward_validate(
        df,
        features,
        target,
        estimator_factory=lambda: build_spread_estimator(league, calibrated=False),
    )

    _print_walk_forward_comparison(calibrated_metrics, uncalibrated_metrics)

    print(f"\n=== HOLDOUT DIAGNOSTIC (SINGLE SPLIT) ===")
    print(f"{'Metric':<20} {'Uncalibrated':<15} {'Calibrated':<15} {'Change':<10}")
    print(f"{'-'*60}")
    print(f"{'Accuracy':<20} {uncal_acc:<15.1%} {cal_acc:<15.1%} {(cal_acc-uncal_acc)*100:+.1f}%")
    print(f"{'Brier Score':<20} {uncal_brier:<15.4f} {cal_brier:<15.4f} {cal_brier-uncal_brier:+.4f}")

    # 11. Calibration analysis - compare predicted vs actual by confidence bucket
    print(f"\n=== HOLDOUT CALIBRATION BY CONFIDENCE BUCKET ===")
    print(f"{'Predicted':<15} {'Actual (uncal)':<18} {'Actual (cal)':<15}")

    for low, high in [(0.50, 0.53), (0.53, 0.55), (0.55, 0.57), (0.57, 0.60), (0.60, 0.65)]:
        # Uncalibrated
        uncal_mask = (uncal_probs >= low) & (uncal_probs < high)
        if uncal_mask.sum() > 0:
            uncal_actual = y_test[uncal_mask].mean()
        else:
            uncal_actual = 0

        # Calibrated
        cal_mask = (cal_probs >= low) & (cal_probs < high)
        if cal_mask.sum() > 0:
            cal_actual = y_test[cal_mask].mean()
        else:
            cal_actual = 0

        mid = (low + high) / 2
        print(f"{mid:.1%}             {uncal_actual:.1%} (n={uncal_mask.sum():<4})    {cal_actual:.1%} (n={cal_mask.sum()})")

    # 12. High confidence accuracy
    high_conf_mask = (cal_probs > 0.53) | (cal_probs < 0.47)
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(y_test[high_conf_mask], cal_preds[high_conf_mask])
        high_conf_count = high_conf_mask.sum()
    else:
        high_conf_acc = 0
        high_conf_count = 0

    print(f"\nHigh Confidence (>53%): {high_conf_acc:.1%} ({high_conf_count} bets)")

    # 13. Feature importance (from base model)
    print(f"\n=== TOP FEATURES ===")
    importance = pd.DataFrame({
        'feature': features,
        'importance': base_clf.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:<20}: {row['importance']:.3f}")

    # 14. Compute sigma for CDF-based line shopping
    # sigma = std(actual_margin - vegas_predicted_margin) where vegas margin = -spread
    # This measures how much actual outcomes deviate from the spread, used to
    # project the classifier's probability at the market spread to other spreads
    # via norm.cdf curves.
    if 'margin' in df_model.columns:
        residuals = df_model['margin'].values - (-df_model['spread'].values)
        sigma = float(np.std(residuals))
        print(f"\nLine shopping sigma: {sigma:.2f} (std of margin vs spread)")
    else:
        sigma = 11.0  # Reasonable CBB default
        print(f"\nLine shopping sigma: {sigma:.2f} (default, margin column not found)")

    # 15. Fit the production model on the full clean dataset, not just the holdout train split.
    production_model = build_spread_estimator(league)
    production_model.fit(df_model[features].astype(float), df_model[target].astype(int))
    joblib.dump({'model': production_model, 'sigma': sigma}, model_file)
    print(f"Production model + sigma saved to {model_file}")


def load_model(path=None, league="mens"):
    """
    Load classifier + sigma from pkl file.

    Handles both old format (raw model) and new format ({'model': ..., 'sigma': ...}).
    Returns (model, sigma).
    """
    if path is None:
        league = normalize_league(league)
        path = get_league_artifact_paths(BASE_DIR, league)["model_file"]

    data = joblib.load(path)
    if isinstance(data, dict) and 'model' in data:
        sigma = data.get('sigma', 11.0)
        return data['model'], float(sigma)
    # Old format: raw model without sigma
    return data, 11.0


def get_feature_list(league="mens"):
    """Return the league-specific spread model feature list."""
    return list(FEATURES_BY_LEAGUE[normalize_league(league)])


def cover_prob_at_spread(classifier_prob, market_spread, alt_spread, sigma):
    """
    Project a classifier's cover probability to a different spread using CDF.

    The classifier gives P(home covers) at the market spread. To get the
    probability at a different spread, we:
    1. Derive an "effective margin" that would produce the classifier's prob
    2. Evaluate the CDF at the alternative spread using that margin

    This guarantees:
    - At market_spread: returns exactly classifier_prob
    - Monotonic: probability increases as spread becomes more favorable
    - Smooth: normal CDF shape between spreads

    Args:
        classifier_prob: P(home covers) from the classifier at market spread
        market_spread: The spread the classifier was evaluated at
        alt_spread: The spread to project to
        sigma: Std dev of (actual_margin - (-spread)) from training data

    Returns:
        P(home covers) at alt_spread
    """
    # Clamp to avoid inf from norm.ppf at 0 or 1
    p = np.clip(classifier_prob, 0.001, 0.999)
    effective_margin = sigma * norm.ppf(p) - market_spread
    return float(norm.cdf((effective_margin + alt_spread) / sigma))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CBB spread model.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League to train: mens or womens (aliases supported).",
    )
    args = parser.parse_args()
    train_and_evaluate(args.league)
