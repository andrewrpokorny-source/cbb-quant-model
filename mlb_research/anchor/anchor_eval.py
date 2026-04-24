"""Three-tier walk-forward evaluation on the frozen MLB anchor.

Reads a model config JSON and runs three independent walk-forward evaluations
on the same frozen data, outputting a single JSON to stdout with three
top-level keys: `optimizer`, `monitor_2025_tail`, `monitor_2026`.

Only the `optimizer` section is the signal the auto-research agent should
optimize against. The two monitor sections exist to detect overfitting to the
optimizer window and MUST NOT drive keep/revert decisions. The runner enforces
that by contract in `program.md`; this script does not hide the columns.

Config schema (example):
    {
        "features": ["is_home", "rest_days", ...],
        "hyperparams": {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "calibration_fraction": 0.2,
            "min_calibration_rows": 200,
            "random_state": 42
        },
        "model_family": "sklearn_gbm",
        "calibrated": true,
        "calibration_method": "sigmoid",
        "target": "home_win"
    }

Supported targets:
  - "home_win" (default): binary classifier; calibrated+sigmoid routes
    through the frozen TimeAwareCalibratedGBM for sklearn_gbm, else through
    the generalized TimeAwareCalibrated wrapper.
  - "margin": regress run margin, convert to P(home wins) via Φ(μ/σ).
    calibrated=true is rejected for this target (separate hypothesis).
    hyperparams.calibration_fraction / min_calibration_rows control the
    residual-std holdout slice.

All fields optional: missing values default to the live production MLB
setup (matches `model.py` / `backtest.py`).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# The anchor is DELIBERATELY self-contained: zero imports from repo-root
# modules. A previous version imported TimeAwareCalibratedGBM from
# /model.py, which meant an edit to that file (e.g. during an unrelated
# production fix) silently shifted anchor scores and invalidated all prior
# rows in results.tsv. The class is frozen here instead. If the live
# production class is improved, the anchor stays pinned to this version
# until the benchmark is deliberately re-frozen.

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANCHOR_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_CSV = os.path.join(ANCHOR_DIR, "mlb_frozen.csv")
MANIFEST_PATH = os.path.join(ANCHOR_DIR, "anchor_manifest.json")


# Frozen copy of production TimeAwareCalibratedGBM as of commit b89f491
# (repo-root /model.py lines 115-196). DO NOT refactor or "improve". Any
# change here breaks comparability with prior experiment rows. If the
# production class diverges, treat that as a DELIBERATE re-freeze event:
# rerun snapshot_data.py --force and reset results.tsv.
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


class TimeAwareCalibrated:
    """Generalized time-aware calibration wrapper.

    Mirrors TimeAwareCalibratedGBM's trailing-window fit/calibrate scheme but:
      - accepts any base estimator via a zero-arg factory (LightGBM, XGBoost,
        sklearn GBM), and
      - supports isotonic calibration in addition to sigmoid (Platt).

    TimeAwareCalibratedGBM stays pinned as a frozen snapshot so the baseline
    row in results.tsv remains bit-reproducible; this class is the code path
    for every other (family, calibration_method) combination.
    """

    def __init__(
        self,
        base_factory: Callable,
        method: str = "sigmoid",
        calibration_fraction: float = 0.2,
        min_calibration_rows: int = 200,
    ):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"method must be 'sigmoid' or 'isotonic', got {method!r}")
        self.base_factory = base_factory
        self.method = method
        self.calibration_fraction = calibration_fraction
        self.min_calibration_rows = min_calibration_rows

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

        self.base_estimator_ = self.base_factory()
        self.base_estimator_.fit(base_X, base_y)
        self.classes_ = np.array([0, 1])

        self.calibrator_ = None
        self.calibration_rows_ = 0
        if use_calibration:
            calib_X = X_df.iloc[split_idx:]
            calib_y = y_ser.iloc[split_idx:]
            raw = np.clip(self.base_estimator_.predict_proba(calib_X)[:, 1], 1e-6, 1 - 1e-6)
            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                cal.fit(raw, calib_y.astype(float).to_numpy())
            else:
                cal = LogisticRegression(solver="lbfgs")
                cal.fit(raw.reshape(-1, 1), calib_y)
            self.calibrator_ = cal
            self.calibration_rows_ = len(calib_X)

        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X).copy()
        raw = np.clip(self.base_estimator_.predict_proba(X_df)[:, 1], 1e-6, 1 - 1e-6)
        if self.calibrator_ is None:
            calibrated = raw
        elif self.method == "isotonic":
            calibrated = np.clip(self.calibrator_.predict(raw), 1e-6, 1 - 1e-6)
        else:
            calibrated = self.calibrator_.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.base_estimator_.feature_importances_


class MarginCDFRegressor:
    """Predict run margin, convert to P(home wins) via Normal CDF.

    Mirrors the CBB CDF-projection trick (memory: `effective_margin = sigma *
    norm.ppf(p) - spread`) but specialized to MLB where the frozen anchor has
    no usable spread (moneyline/run_line are 100% NaN). P(home wins) = Φ(μ/σ)
    where μ is the predicted margin and σ is the residual standard deviation
    estimated on a trailing holdout slice (same split discipline as the
    sigmoid/isotonic calibration wrappers, so σ is not an in-sample
    underestimate).

    Exposes a classifier-shaped interface (predict_proba, predict, classes_)
    so the walk_forward_window driver does not need to special-case it.
    """

    def __init__(
        self,
        base_factory: Callable,
        residual_fraction: float = 0.2,
        min_residual_rows: int = 200,
        min_sigma: float = 0.5,
    ):
        self.base_factory = base_factory
        self.residual_fraction = residual_fraction
        self.min_residual_rows = min_residual_rows
        self.min_sigma = min_sigma

    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if len(y_arr) != len(X_df):
            raise ValueError("X and y length mismatch")
        self.feature_names_in_ = X_df.columns.astype(str).to_numpy()
        self.n_features_in_ = len(self.feature_names_in_)

        split_idx = max(1, int(len(X_df) * (1 - self.residual_fraction)))
        split_idx = min(split_idx, len(X_df) - 1)

        use_holdout = len(X_df) >= self.min_residual_rows and split_idx < len(X_df)

        base_X = X_df.iloc[:split_idx] if use_holdout else X_df
        base_y = y_arr[:split_idx] if use_holdout else y_arr

        self.base_estimator_ = self.base_factory()
        self.base_estimator_.fit(base_X, base_y)
        self.classes_ = np.array([0, 1])

        if use_holdout:
            held_X = X_df.iloc[split_idx:]
            held_y = y_arr[split_idx:]
            residuals = held_y - self.base_estimator_.predict(held_X)
            self.residual_rows_ = len(held_X)
        else:
            residuals = y_arr - self.base_estimator_.predict(X_df)
            self.residual_rows_ = 0

        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
        if not np.isfinite(sigma) or sigma < self.min_sigma:
            sigma = self.min_sigma
        self.sigma_ = sigma
        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X).copy()
        mu = np.asarray(self.base_estimator_.predict(X_df), dtype=float)
        p_home = norm.cdf(mu / self.sigma_)
        p_home = np.clip(p_home, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p_home, p_home])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.base_estimator_.feature_importances_


# Pinned at the harness level. Configs are NOT allowed to override this --
# a per-experiment threshold knob makes opt_roi non-comparable across rows
# (a config with threshold=0.99 trivially gets n_hc=0 and roi=None).
HIGH_CONF_THRESHOLD = 0.53
MIN_TRAIN_ROWS = 50

# Production-default feature list for MLB (mirrors MLB_FEATURES in model.py).
# Duplicated here so the anchor is self-contained: if the agent edits the
# upstream MLB_FEATURES list during an experiment, we do not want the anchor's
# default silently shifting.
DEFAULT_MLB_FEATURES = [
    "is_home",
    "rest_days",
    "sp_era",
    "opp_sp_era",
    "sp_roll_era",
    "sp_roll_whip",
    "sp_roll_k9",
    "sp_roll_ip",
    "opp_sp_roll_era",
    "prev_roll10_runs_per_game",
    "prev_roll10_runs_allowed",
    "prev_season_runs_per_game",
    "prev_season_runs_allowed",
    "prev_games_played",
    "opp_win_pct",
    "prev_win_pct",
    "prev_roll10_win_pct",
    "roll10_rpg_diff",
    "roll10_ra_diff",
    "sp_era_diff",
    "sp_roll_era_diff",
    "prev_volatility",
    "prev_season_pyth_wpct",
    "prev_roll10_pyth_wpct",
    "pyth_wpct_diff",
    "wind_speed",
    "bullpen_era_diff",
    "roll5_rpg_diff",
]
DEFAULT_HYPERPARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "calibration_fraction": 0.2,
    "min_calibration_rows": 200,
    "random_state": 42,
}


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _sha256_of_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_df() -> pd.DataFrame:
    """Load the frozen CSV, asserting its SHA256 matches the manifest.

    The 444 permission is an honor system; a malicious or buggy agent can
    chmod + rewrite the file. Re-hashing on every eval is cheap (~30ms for
    5 MB) and closes the tampering + corruption paths in one shot.
    """
    manifest = load_manifest()
    actual = _sha256_of_file(FROZEN_CSV)
    expected = manifest["sha256"]
    if actual != expected:
        raise RuntimeError(
            f"Frozen CSV SHA256 mismatch: expected {expected}, got {actual}. "
            "The anchor has been modified since snapshot. Refusing to evaluate."
        )

    df = pd.read_csv(FROZEN_CSV, low_memory=False)
    if len(df) != manifest["row_count"]:
        raise RuntimeError(
            f"Frozen CSV row count mismatch: expected {manifest['row_count']}, "
            f"got {len(df)}."
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Mirror backtest.py: derive rest_days from scratch so it is consistent
    # regardless of what the upstream CSV contains.
    df["_last_game"] = df.groupby("team")["date"].shift(1)
    df["rest_days"] = (df["date"] - df["_last_game"]).dt.days.fillna(7).clip(upper=7)
    df = df.drop(columns=["_last_game"])
    return df


SUPPORTED_MODEL_FAMILIES = {"sklearn_gbm", "lightgbm", "xgboost"}
SUPPORTED_CALIBRATION_METHODS = {"sigmoid", "isotonic"}
SUPPORTED_TARGETS = {"home_win", "margin"}


def build_estimator_factory(config: dict) -> Callable:
    hp = {**DEFAULT_HYPERPARAMS, **(config.get("hyperparams") or {})}
    calibrated = config.get("calibrated", True)
    family = config.get("model_family", "sklearn_gbm")
    method = config.get("calibration_method", "sigmoid")
    target = config.get("target", "home_win")
    if family not in SUPPORTED_MODEL_FAMILIES:
        sys.exit(
            f"Unsupported model_family={family!r}. "
            f"Must be one of {sorted(SUPPORTED_MODEL_FAMILIES)}."
        )
    if method not in SUPPORTED_CALIBRATION_METHODS:
        sys.exit(
            f"Unsupported calibration_method={method!r}. "
            f"Must be one of {sorted(SUPPORTED_CALIBRATION_METHODS)}."
        )
    if target not in SUPPORTED_TARGETS:
        sys.exit(
            f"Unsupported target={target!r}. Must be one of {sorted(SUPPORTED_TARGETS)}."
        )
    if target == "margin" and calibrated:
        sys.exit(
            "calibrated=true is not supported with target='margin'. The margin "
            "path already produces a Φ(μ/σ) probability; post-hoc calibration "
            "on top is a separate hypothesis. Set calibrated=false."
        )

    def _build_base_classifier(hp_dict, fam):
        if fam == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=hp_dict["n_estimators"],
                learning_rate=hp_dict["learning_rate"],
                max_depth=hp_dict["max_depth"],
                random_state=hp_dict["random_state"],
                subsample=hp_dict.get("subsample", 0.8),
                colsample_bytree=hp_dict.get("colsample_bytree", 0.8),
                verbosity=-1,
            )
        if fam == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=hp_dict["n_estimators"],
                learning_rate=hp_dict["learning_rate"],
                max_depth=hp_dict["max_depth"],
                random_state=hp_dict["random_state"],
                subsample=hp_dict.get("subsample", 0.8),
                colsample_bytree=hp_dict.get("colsample_bytree", 0.8),
                eval_metric="logloss",
                verbosity=0,
            )
        return GradientBoostingClassifier(
            n_estimators=hp_dict["n_estimators"],
            learning_rate=hp_dict["learning_rate"],
            max_depth=hp_dict["max_depth"],
            random_state=hp_dict["random_state"],
        )

    def _build_base_regressor(hp_dict, fam):
        if fam == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(
                n_estimators=hp_dict["n_estimators"],
                learning_rate=hp_dict["learning_rate"],
                max_depth=hp_dict["max_depth"],
                random_state=hp_dict["random_state"],
                subsample=hp_dict.get("subsample", 0.8),
                colsample_bytree=hp_dict.get("colsample_bytree", 0.8),
                verbosity=-1,
            )
        if fam == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=hp_dict["n_estimators"],
                learning_rate=hp_dict["learning_rate"],
                max_depth=hp_dict["max_depth"],
                random_state=hp_dict["random_state"],
                subsample=hp_dict.get("subsample", 0.8),
                colsample_bytree=hp_dict.get("colsample_bytree", 0.8),
                verbosity=0,
            )
        return GradientBoostingRegressor(
            n_estimators=hp_dict["n_estimators"],
            learning_rate=hp_dict["learning_rate"],
            max_depth=hp_dict["max_depth"],
            random_state=hp_dict["random_state"],
        )

    def factory():
        if target == "margin":
            return MarginCDFRegressor(
                base_factory=lambda: _build_base_regressor(hp, family),
                residual_fraction=hp["calibration_fraction"],
                min_residual_rows=hp["min_calibration_rows"],
            )
        # Baseline path: sklearn_gbm + calibrated + sigmoid routes through the
        # frozen TimeAwareCalibratedGBM so the baseline row stays bit-reproducible.
        if calibrated and family == "sklearn_gbm" and method == "sigmoid":
            return TimeAwareCalibratedGBM(
                n_estimators=hp["n_estimators"],
                learning_rate=hp["learning_rate"],
                max_depth=hp["max_depth"],
                random_state=hp["random_state"],
                calibration_fraction=hp["calibration_fraction"],
                min_calibration_rows=hp["min_calibration_rows"],
            )
        if calibrated:
            return TimeAwareCalibrated(
                base_factory=lambda: _build_base_classifier(hp, family),
                method=method,
                calibration_fraction=hp["calibration_fraction"],
                min_calibration_rows=hp["min_calibration_rows"],
            )
        return _build_base_classifier(hp, family)

    return factory


EVAL_TARGET = "home_win"


def walk_forward_window(
    df: pd.DataFrame,
    features: list,
    target: str,
    window_start: datetime,
    window_end: datetime,
    estimator_factory: Callable,
) -> tuple[pd.DataFrame | None, dict]:
    """Walk-forward over a single date window.

    Trains on all games in `df` strictly before each weekly cutoff; tests on
    home-team rows within that week (home-only to avoid double-counting each
    game). Returns (per-prediction DataFrame, diagnostics). Diagnostics
    include train-row sizes per fold and skip counts -- these surface silent
    row-drops from ``dropna`` when a new feature has coverage gaps.

    The training target (`target`) may be either the binary `home_win`
    (classifier path) or a continuous outcome like `margin` (regressor-plus-
    CDF path). Evaluation is ALWAYS scored against the binary `home_win`
    column so Brier / ROI / n_hc metrics stay comparable across targets.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Config references features not in frozen CSV: {missing}")
    if target not in df.columns:
        raise ValueError(f"Training target {target!r} not in frozen CSV.")
    if EVAL_TARGET not in df.columns:
        raise ValueError(f"Evaluation target {EVAL_TARGET!r} not in frozen CSV.")

    is_regression_target = target != EVAL_TARGET
    dropna_cols = features + [target]
    if is_regression_target:
        dropna_cols = dropna_cols + [EVAL_TARGET]

    fold_logs = []
    train_sizes = []
    skipped_thin_train = 0
    skipped_empty_week = 0

    current = window_start
    while current < window_end:
        next_week = current + timedelta(days=7)

        train = df[df["date"] < current].dropna(subset=dropna_cols)
        if len(train) < MIN_TRAIN_ROWS:
            skipped_thin_train += 1
            current = next_week
            continue

        week_mask = (
            (df["date"] >= current)
            & (df["date"] < next_week)
            & (df["is_home"] == 1)
        )
        week = df.loc[week_mask].dropna(subset=dropna_cols)
        if week.empty:
            skipped_empty_week += 1
            current = next_week
            continue

        est = estimator_factory()
        if is_regression_target:
            est.fit(train[features].astype(float), train[target].astype(float))
        else:
            est.fit(train[features].astype(float), train[target].astype(int))
        probs = est.predict_proba(week[features].astype(float))[:, 1]

        train_sizes.append(int(len(train)))
        fold_logs.append(
            pd.DataFrame(
                {
                    "date": week["date"].values,
                    "prob_home": probs,
                    "target": week[EVAL_TARGET].astype(int).values,
                    "conf": np.maximum(probs, 1 - probs),
                }
            )
        )
        current = next_week

    diagnostics = {
        "n_folds_trained": len(fold_logs),
        "n_folds_skipped_thin_train": skipped_thin_train,
        "n_folds_skipped_empty_week": skipped_empty_week,
        "train_rows_min": min(train_sizes) if train_sizes else None,
        "train_rows_max": max(train_sizes) if train_sizes else None,
        "train_rows_mean": (sum(train_sizes) / len(train_sizes)) if train_sizes else None,
    }
    if not fold_logs:
        return None, diagnostics
    return pd.concat(fold_logs, ignore_index=True), diagnostics


def summarize(predictions: pd.DataFrame | None, high_conf_threshold: float) -> dict:
    if predictions is None or predictions.empty:
        return {
            "brier": None,
            "log_loss": None,
            "accuracy": None,
            "high_conf_accuracy": None,
            "roi_units": None,
            "n_games": 0,
            "n_high_conf": 0,
        }

    y = predictions["target"].astype(int).to_numpy()
    p = predictions["prob_home"].astype(float).to_numpy()
    pred_class = (p > 0.5).astype(int)

    brier = float(brier_score_loss(y, p))
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(log_loss(y, p_clipped, labels=[0, 1]))
    acc = float((pred_class == y).mean())

    hc_mask = predictions["conf"] >= high_conf_threshold
    n_hc = int(hc_mask.sum())
    if n_hc:
        hc_correct = int((pred_class[hc_mask.values] == y[hc_mask.values]).sum())
        hc_acc = hc_correct / n_hc
        payout = 100.0 / 110.0  # -110 break-even payout
        roi = float((hc_correct * payout) - (n_hc - hc_correct))
    else:
        # Zero high-conf picks: ROI is undefined, not zero. A `0.0` here would
        # be indistinguishable from a break-even 54-pick slate and would fool
        # the keep/revert rule.
        hc_acc = None
        roi = None

    return {
        "brier": brier,
        "log_loss": ll,
        "accuracy": acc,
        "high_conf_accuracy": hc_acc,
        "roi_units": roi,
        "n_games": int(len(predictions)),
        "n_high_conf": n_hc,
    }


def parse_window_bounds(manifest: dict, key: str) -> tuple[datetime, datetime]:
    w = manifest["windows"][key]
    # end is inclusive in the manifest; walk_forward_window treats end as
    # exclusive-upper so add a day.
    start = datetime.fromisoformat(w["start"])
    end_inclusive = datetime.fromisoformat(w["end"])
    return start, end_inclusive + timedelta(days=1)


def atomic_write_json(path: str, obj: dict):
    """Write JSON atomically so a crash mid-write cannot leave a half-file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model config JSON. Defaults to production MLB setup.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "If given, write results JSON atomically to this path (runner "
            "uses this to avoid parsing stdout which can be corrupted by "
            "stray prints from imported modules). If omitted, print to "
            "stdout for interactive use."
        ),
    )
    args = parser.parse_args()

    if args.model_config:
        with open(args.model_config) as f:
            config = json.load(f)
    else:
        config = {}

    if "high_conf_threshold" in config:
        sys.exit(
            "Config must not set 'high_conf_threshold'. The harness pins "
            f"it at {HIGH_CONF_THRESHOLD} so opt_roi is comparable across "
            "experiments. Remove the key and try again."
        )

    features = config.get("features") or DEFAULT_MLB_FEATURES
    target = config.get("target", "home_win")

    manifest = load_manifest()
    df = load_frozen_df()
    factory = build_estimator_factory(config)

    results = {}
    diagnostics_by_window = {}
    for key in ("optimizer", "monitor_2025_tail", "monitor_2026"):
        start, end_exclusive = parse_window_bounds(manifest, key)
        preds, diag = walk_forward_window(df, features, target, start, end_exclusive, factory)
        results[key] = summarize(preds, HIGH_CONF_THRESHOLD)
        diagnostics_by_window[key] = diag

    results["_meta"] = {
        "features_used": features,
        "n_features": len(features),
        "model_family": config.get("model_family", "sklearn_gbm"),
        "calibrated": config.get("calibrated", True),
        "calibration_method": config.get("calibration_method", "sigmoid"),
        "target": target,
        "high_conf_threshold": HIGH_CONF_THRESHOLD,
        "anchor_sha256": manifest["sha256"],
        "hyperparams": {**DEFAULT_HYPERPARAMS, **(config.get("hyperparams") or {})},
        "diagnostics": diagnostics_by_window,
    }

    if args.output:
        atomic_write_json(args.output, results)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
