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
FROZEN_CSV = os.environ.get("MLB_RESEARCH_FROZEN_CSV") or os.path.join(
    ANCHOR_DIR, "mlb_frozen.csv"
)
MANIFEST_PATH = os.environ.get("MLB_RESEARCH_ANCHOR_MANIFEST") or os.path.join(
    ANCHOR_DIR, "anchor_manifest.json"
)

sys.path.insert(0, REPO_ROOT)
from mlb_research.market_odds import american_odds_profit  # noqa: E402


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

        # Gate intentionally MATCHES the frozen TimeAwareCalibratedGBM: total
        # rows >= min_calibration_rows + class-balance on both halves. An
        # earlier draft added a min_holdout_rows floor here, but adversarial
        # review pointed out that diverging from the frozen class's gate
        # made cross-family comparisons against the baseline confounded by
        # wrapper policy. Until a deliberate re-baseline is done, both paths
        # use the same legacy policy. The thin-holdout calibrator concern is
        # real but inherited from the baseline -- it affects every row
        # equally, so optimizer deltas are still apples-to-apples.
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
        self.calibrator_source_ = "skipped_thin_holdout"
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
            self.calibrator_source_ = "holdout"

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

        # Gate INTENTIONALLY mirrors the calibration wrapper's policy (which
        # in turn mirrors the frozen TimeAwareCalibratedGBM). When the gate
        # rejects a fold, sigma falls back to std-of-y with a confidence
        # clamp in predict_proba so no high-conf picks come out of those
        # folds. The earlier min_holdout_rows floor was removed for cross-
        # path comparison fairness; right resolution is a deliberate
        # re-baseline.
        use_holdout = (
            len(X_df) >= self.min_residual_rows
            and split_idx < len(X_df)
        )

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
            self.sigma_source_ = "holdout"
            sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
        else:
            # Conservative fallback: std of training y itself. This is the
            # variance of an unconditional model -- strictly >= the residual
            # std of any honest out-of-sample fit -- so probabilities will
            # collapse toward 0.5 in thin folds rather than spiking on the
            # in-sample-residual underestimate that tree ensembles produce.
            self.residual_rows_ = 0
            self.sigma_source_ = "std_of_y_fallback"
            sigma = float(np.std(y_arr, ddof=1)) if len(y_arr) > 1 else float("nan")

        if not np.isfinite(sigma) or sigma < self.min_sigma:
            sigma = self.min_sigma
        self.sigma_ = sigma
        return self

    def predict_proba(self, X):
        X_df = pd.DataFrame(X).copy()
        mu = np.asarray(self.base_estimator_.predict(X_df), dtype=float)
        z = mu / self.sigma_
        if self.sigma_source_ == "std_of_y_fallback":
            # std(y) is NOT a guaranteed upper bound on out-of-sample residual
            # variance: a misspecified or overfit regressor can have residuals
            # exceeding the unconditional target std. Adversarial review
            # caught norm.cdf(mu/std(y)) producing inflated extreme probs on
            # exactly the thin folds the fallback was meant to protect.
            #
            # Treatment: clamp |z| to a tight band so confidence stays under
            # any reasonable HIGH_CONF_THRESHOLD. Brier on these folds will
            # be near 0.25 (Brier(0.5, y)) and n_high_conf contribution is 0,
            # which honestly reflects "we have no honest sigma estimate here".
            # Ordinal information from mu is preserved at the third decimal
            # so Brier still has a tiny gradient if the underlying mu
            # actually correlates with outcomes.
            z = np.clip(z, -0.005, 0.005)
        p_home = norm.cdf(z)
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
# Family-specific defaults that the estimator builders silently apply when
# the config omits these keys. They ARE active hyperparameters of the fitted
# model, so they must appear in _meta.hyperparams to keep the archive
# truthful (caught by adversarial review). sklearn_gbm has no extra defaults.
FAMILY_DEFAULT_HYPERPARAMS = {
    "lightgbm": {"subsample": 0.8, "colsample_bytree": 0.8},
    "xgboost": {"subsample": 0.8, "colsample_bytree": 0.8},
    "sklearn_gbm": {},
}


def effective_hyperparams(config: dict) -> dict:
    """Return the hyperparam map actually applied to the fitted model.

    Layer order: framework defaults < family-specific defaults < config
    overrides. Then filtered to only keys that are active under the
    chosen path, so archived rows don't claim keys that the estimator
    silently ignored (sklearn_gbm dropping `subsample`, etc.).
    """
    family = config.get("model_family", "sklearn_gbm")
    merged = {
        **DEFAULT_HYPERPARAMS,
        **FAMILY_DEFAULT_HYPERPARAMS.get(family, {}),
        **(config.get("hyperparams") or {}),
    }
    active = active_hyperparam_keys(config)
    return {k: v for k, v in merged.items() if k in active}


def load_manifest() -> dict:
    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Anchor manifest not found at {MANIFEST_PATH!r}. "
            "Check MLB_RESEARCH_ANCHOR_MANIFEST."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Anchor manifest at {MANIFEST_PATH!r} is not valid JSON."
        ) from exc

    missing = [key for key in ("sha256", "row_count", "windows") if key not in manifest]
    if missing:
        raise RuntimeError(
            f"Anchor manifest at {MANIFEST_PATH!r} missing required key(s): {missing}."
        )
    return manifest


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
SUPPORTED_ROI_MODES = {"flat_110", "moneyline"}


def calibration_method_is_active(config: dict) -> bool:
    """Return True iff `calibration_method` actually affects the execution path.

    A `calibration_method` value only does something when the classifier path
    is taken WITH calibration on top. If `calibrated=false` or `target='margin'`,
    no calibrator is ever fit, so any `calibration_method` setting is inert and
    must not be silently recorded as if it had been applied.
    """
    return (
        config.get("target", "home_win") == "home_win"
        and bool(config.get("calibrated", True))
    )


def build_estimator_factory(config: dict) -> Callable:
    hp = {**DEFAULT_HYPERPARAMS, **(config.get("hyperparams") or {})}
    calibrated = config.get("calibrated", True)
    family = config.get("model_family", "sklearn_gbm")
    method_explicitly_set = "calibration_method" in config
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
    # Reject inert `calibration_method` so the archived ledger label cannot lie
    # about which mechanism was exercised. Caught by adversarial review pre-Run-3.
    if method_explicitly_set and not calibration_method_is_active(config):
        if not calibrated:
            sys.exit(
                "calibration_method is set but calibrated=false. The method "
                "key has no effect when calibration is disabled, so recording "
                "it would mis-label the experiment. Remove calibration_method "
                "or set calibrated=true."
            )
        if target == "margin":
            sys.exit(
                "calibration_method is set but target='margin'. The margin "
                "path does not apply post-hoc calibration, so recording the "
                "method would mis-label the experiment. Remove calibration_method."
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
    calibrator_source_counts: dict[str, int] = {}
    sigma_source_counts: dict[str, int] = {}

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
        home_moneyline = (
            week["team_moneyline"]
            if "team_moneyline" in week.columns
            else week["moneyline"]
            if "moneyline" in week.columns
            else pd.Series(np.nan, index=week.index)
        )
        away_moneyline = (
            week["opp_moneyline"]
            if "opp_moneyline" in week.columns
            else pd.Series(np.nan, index=week.index)
        )

        # Surface per-fold fallback usage so the metrics JSON makes silent
        # path-fallback observable. The new TimeAwareCalibrated wrapper
        # sets calibrator_source_ explicitly. The frozen
        # TimeAwareCalibratedGBM does not -- derive the source for it (and
        # any other wrapper that exposes calibrator_) from whether
        # calibrator_ was actually fit. Raw classifiers/regressors lack
        # calibrator_ entirely so the fold is correctly not counted, which
        # keeps uncalibrated runs out of the fallback-share gate.
        cal_src = getattr(est, "calibrator_source_", None)
        if cal_src is None and hasattr(est, "calibrator_"):
            cal_src = "holdout" if est.calibrator_ is not None else "skipped_thin_holdout"
        if cal_src is not None:
            calibrator_source_counts[cal_src] = calibrator_source_counts.get(cal_src, 0) + 1
        sig_src = getattr(est, "sigma_source_", None)
        if sig_src is not None:
            sigma_source_counts[sig_src] = sigma_source_counts.get(sig_src, 0) + 1

        train_sizes.append(int(len(train)))
        fold_logs.append(
            pd.DataFrame(
                {
                    "date": week["date"].values,
                    "prob_home": probs,
                    "target": week[EVAL_TARGET].astype(int).values,
                    "conf": np.maximum(probs, 1 - probs),
                    "home_moneyline": home_moneyline.values,
                    "away_moneyline": away_moneyline.values,
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
        "calibrator_source_counts": calibrator_source_counts,
        "sigma_source_counts": sigma_source_counts,
    }
    if not fold_logs:
        return None, diagnostics
    return pd.concat(fold_logs, ignore_index=True), diagnostics


def _moneyline_roi_units(
    predictions: pd.DataFrame,
    hc_mask: pd.Series,
    pred_class: np.ndarray,
    y: np.ndarray,
) -> tuple[float | None, int, int]:
    """Score high-confidence picks at the selected side's actual moneyline."""
    roi = 0.0
    priced = 0
    missing = 0
    for pos, is_hc in enumerate(hc_mask.to_numpy()):
        if not is_hc:
            continue
        selected_odds = (
            predictions.iloc[pos]["home_moneyline"]
            if pred_class[pos] == 1
            else predictions.iloc[pos]["away_moneyline"]
        )
        profit = american_odds_profit(selected_odds)
        if not np.isfinite(profit):
            missing += 1
            continue
        priced += 1
        roi += profit if pred_class[pos] == y[pos] else -1.0
    return (float(roi) if priced else None), priced, missing


def summarize(
    predictions: pd.DataFrame | None,
    high_conf_threshold: float,
    roi_mode: str = "flat_110",
) -> dict:
    if predictions is None or predictions.empty:
        return {
            "brier": None,
            "log_loss": None,
            "accuracy": None,
            "high_conf_accuracy": None,
            "roi_units": None,
            "roi_mode": roi_mode,
            "n_roi_priced": 0,
            "n_roi_missing_price": 0,
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
        if roi_mode == "moneyline":
            roi, n_roi_priced, n_roi_missing = _moneyline_roi_units(
                predictions, hc_mask, pred_class, y
            )
        else:
            payout = 100.0 / 110.0  # -110 break-even payout
            roi = float((hc_correct * payout) - (n_hc - hc_correct))
            n_roi_priced = n_hc
            n_roi_missing = 0
    else:
        # Zero high-conf picks: ROI is undefined, not zero. A `0.0` here would
        # be indistinguishable from a break-even 54-pick slate and would fool
        # the keep/revert rule.
        hc_acc = None
        roi = None
        n_roi_priced = 0
        n_roi_missing = 0

    return {
        "brier": brier,
        "log_loss": ll,
        "accuracy": acc,
        "high_conf_accuracy": hc_acc,
        "roi_units": roi,
        "roi_mode": roi_mode,
        "n_roi_priced": n_roi_priced,
        "n_roi_missing_price": n_roi_missing,
        "n_games": int(len(predictions)),
        "n_high_conf": n_hc,
    }


def parse_window_bounds(manifest: dict, key: str) -> tuple[datetime, datetime]:
    try:
        w = manifest["windows"][key]
        start_raw = w["start"]
        end_raw = w["end"]
    except KeyError as exc:
        windows = manifest.get("windows") if isinstance(manifest, dict) else None
        available = sorted(windows) if isinstance(windows, dict) else []
        raise RuntimeError(
            f"Anchor manifest at {MANIFEST_PATH!r} missing window bounds for "
            f"{key!r}; available windows: {available}."
        ) from exc
    # end is inclusive in the manifest; walk_forward_window treats end as
    # exclusive-upper so add a day.
    start = datetime.fromisoformat(start_raw)
    end_inclusive = datetime.fromisoformat(end_raw)
    return start, end_inclusive + timedelta(days=1)


def atomic_write_json(path: str, obj: dict):
    """Write JSON atomically so a crash mid-write cannot leave a half-file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


VALID_TOP_LEVEL_CONFIG_KEYS = {
    "features",
    "hyperparams",
    "model_family",
    "calibrated",
    "calibration_method",
    "target",
    "roi_mode",
}

VALID_HYPERPARAM_KEYS = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "calibration_fraction",
    "min_calibration_rows",
    "random_state",
    "subsample",
    "colsample_bytree",
}

# Hyperparameter activeness rules. Each key is "active" only when the chosen
# (model_family, target, calibrated) combo actually plumbs it through to an
# estimator or wrapper. An inert hyperparameter recorded in the ledger would
# falsely advertise that the experiment tested that knob -- caught by
# adversarial review pre-Run-3.
_CORE_HYPERPARAM_KEYS = {"n_estimators", "max_depth", "learning_rate", "random_state"}
_SAMPLING_HYPERPARAM_KEYS = {"subsample", "colsample_bytree"}
_HOLDOUT_HYPERPARAM_KEYS = {"calibration_fraction", "min_calibration_rows"}


def active_hyperparam_keys(config: dict) -> set:
    """Return the subset of VALID_HYPERPARAM_KEYS that the chosen path actually uses.

    - Core keys (n_estimators / max_depth / learning_rate / random_state) are
      always active across all model families and targets.
    - Sampling keys (subsample / colsample_bytree) are only plumbed through
      to LightGBM / XGBoost; sklearn_gbm silently ignores them.
    - Holdout keys (calibration_fraction / min_calibration_rows) drive the
      trailing-window slice for either calibration (when calibrated=true on
      home_win) or residual sigma estimation (when target=margin). Inert
      otherwise.
    """
    family = config.get("model_family", "sklearn_gbm")
    calibrated = bool(config.get("calibrated", True))
    target = config.get("target", "home_win")

    keys = set(_CORE_HYPERPARAM_KEYS)
    if family in {"lightgbm", "xgboost"}:
        keys |= _SAMPLING_HYPERPARAM_KEYS
    calib_active = target == "home_win" and calibrated
    margin_active = target == "margin"
    if calib_active or margin_active:
        keys |= _HOLDOUT_HYPERPARAM_KEYS
    return keys


_INT_HYPERPARAMS = {"n_estimators", "max_depth", "random_state", "min_calibration_rows"}
_FLOAT_HYPERPARAMS = {
    "learning_rate",
    "calibration_fraction",
    "subsample",
    "colsample_bytree",
}


def _validate_config_types(config: dict):
    """Reject malformed types before activeness/routing.

    Adversarial review caught: `{"calibrated": "false"}` previously passed
    validation, was treated as truthy by Python, and ran the calibrated path
    while `_meta` claimed otherwise. JSON makes string-bool typos easy in
    autonomous loops, so type checks happen before any path decision.
    """
    if "calibrated" in config and not isinstance(config["calibrated"], bool):
        sys.exit(
            "Config field `calibrated` must be a JSON boolean (true/false), "
            f"got {type(config['calibrated']).__name__}: {config['calibrated']!r}. "
            "A string like \"false\" is truthy in Python and would silently "
            "route the calibrated path."
        )

    if "features" in config and not isinstance(config["features"], list):
        sys.exit(
            f"Config field `features` must be a JSON array, got "
            f"{type(config['features']).__name__}."
        )

    hp = config.get("hyperparams")
    if hp is not None and not isinstance(hp, dict):
        sys.exit(
            f"Config field `hyperparams` must be a JSON object, got "
            f"{type(hp).__name__}."
        )
    if isinstance(hp, dict):
        for k, v in hp.items():
            if k.startswith("_"):
                continue
            # bool is a subclass of int in Python, so reject explicitly.
            if isinstance(v, bool):
                sys.exit(
                    f"hyperparams.{k} must be numeric, got bool {v!r}. "
                    "JSON booleans are not valid hyperparameter values."
                )
            if k in _INT_HYPERPARAMS and not isinstance(v, int):
                sys.exit(
                    f"hyperparams.{k} must be an integer, got "
                    f"{type(v).__name__}: {v!r}."
                )
            if k in _FLOAT_HYPERPARAMS and not isinstance(v, (int, float)):
                sys.exit(
                    f"hyperparams.{k} must be a number, got "
                    f"{type(v).__name__}: {v!r}."
                )


def validate_config_keys(config: dict):
    """Reject malformed configs before they reach build_estimator_factory.

    Validation layers:
      1. Field types (calibrated must be bool, hyperparams must be dict, etc.)
      2. Unknown top-level keys (typos like `model_familyy`).
      3. Unknown hyperparams keys (typos like `n_estimator`).
      4. Inert hyperparams for the chosen path (e.g. `subsample` on
         sklearn_gbm, or `calibration_fraction` on uncalibrated home_win).

    Keys starting with `_` are treated as comments by convention.
    """
    _validate_config_types(config)

    unknown_top = [
        k for k in config
        if not k.startswith("_") and k not in VALID_TOP_LEVEL_CONFIG_KEYS
    ]
    if unknown_top:
        sys.exit(
            f"Unknown top-level config key(s): {unknown_top}. "
            f"Valid keys: {sorted(VALID_TOP_LEVEL_CONFIG_KEYS)} "
            "(keys starting with `_` are treated as comments)."
        )
    hp = config.get("hyperparams") or {}
    unknown_hp = [
        k for k in hp
        if not k.startswith("_") and k not in VALID_HYPERPARAM_KEYS
    ]
    if unknown_hp:
        sys.exit(
            f"Unknown hyperparams key(s): {unknown_hp}. "
            f"Valid keys: {sorted(VALID_HYPERPARAM_KEYS)}."
        )
    active = active_hyperparam_keys(config)
    inert_hp = sorted(
        k for k in hp
        if not k.startswith("_") and k in VALID_HYPERPARAM_KEYS and k not in active
    )
    if inert_hp:
        family = config.get("model_family", "sklearn_gbm")
        calibrated = bool(config.get("calibrated", True))
        target = config.get("target", "home_win")
        sys.exit(
            f"Inert hyperparams for (model_family={family}, target={target}, "
            f"calibrated={calibrated}): {inert_hp}. These keys are not plumbed "
            "through to the active estimator/wrapper, so recording them would "
            "mis-label the experiment. Remove them, or change "
            "model_family/target/calibrated to a combo where they are active."
        )


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

    validate_config_keys(config)

    features = config.get("features") or DEFAULT_MLB_FEATURES
    target = config.get("target", "home_win")
    roi_mode = config.get("roi_mode", "flat_110")
    if roi_mode not in SUPPORTED_ROI_MODES:
        sys.exit(
            f"Unsupported roi_mode={roi_mode!r}. Must be one of "
            f"{sorted(SUPPORTED_ROI_MODES)}."
        )

    manifest = load_manifest()
    df = load_frozen_df()
    factory = build_estimator_factory(config)

    results = {}
    diagnostics_by_window = {}
    for key in ("optimizer", "monitor_2025_tail", "monitor_2026"):
        start, end_exclusive = parse_window_bounds(manifest, key)
        preds, diag = walk_forward_window(df, features, target, start, end_exclusive, factory)
        results[key] = summarize(preds, HIGH_CONF_THRESHOLD, roi_mode=roi_mode)
        diagnostics_by_window[key] = diag

    results["_meta"] = {
        "features_used": features,
        "n_features": len(features),
        "model_family": config.get("model_family", "sklearn_gbm"),
        "calibrated": config.get("calibrated", True),
        # Emit calibration_method ONLY when it can actually affect execution.
        # Recording "isotonic" on a calibrated=false or target=margin run
        # mis-labels the experiment (the path didn't exercise it).
        "calibration_method": (
            config.get("calibration_method", "sigmoid")
            if calibration_method_is_active(config)
            else None
        ),
        "target": target,
        "roi_mode": roi_mode,
        "high_conf_threshold": HIGH_CONF_THRESHOLD,
        "anchor_sha256": manifest["sha256"],
        # Emit family-aware effective hyperparameters: framework defaults
        # plus family defaults (subsample/colsample_bytree for LGBM/XGB)
        # plus config overrides, filtered to active keys. Recording the
        # actually-applied 0.8 sampling defaults for LGBM was missed by
        # an earlier draft -- caught by adversarial review.
        "hyperparams": effective_hyperparams(config),
        "diagnostics": diagnostics_by_window,
    }

    if args.output:
        atomic_write_json(args.output, results)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
