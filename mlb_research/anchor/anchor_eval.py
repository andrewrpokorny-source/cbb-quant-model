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
        "calibrated": true,
        "target": "home_win",
        "high_conf_threshold": 0.53
    }

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
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

# Import from repo root so we reuse the production estimator class.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from model import TimeAwareCalibratedGBM  # noqa: E402

ANCHOR_DIR = os.path.dirname(os.path.abspath(__file__))
FROZEN_CSV = os.path.join(ANCHOR_DIR, "mlb_frozen.csv")
MANIFEST_PATH = os.path.join(ANCHOR_DIR, "anchor_manifest.json")

HIGH_CONF_THRESHOLD_DEFAULT = 0.53
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


def load_frozen_df() -> pd.DataFrame:
    df = pd.read_csv(FROZEN_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Mirror backtest.py: derive rest_days from scratch so it is consistent
    # regardless of what the upstream CSV contains.
    df["_last_game"] = df.groupby("team")["date"].shift(1)
    df["rest_days"] = (df["date"] - df["_last_game"]).dt.days.fillna(7).clip(upper=7)
    df = df.drop(columns=["_last_game"])
    return df


def build_estimator_factory(config: dict) -> Callable:
    hp = {**DEFAULT_HYPERPARAMS, **(config.get("hyperparams") or {})}
    calibrated = config.get("calibrated", True)

    def factory():
        if calibrated:
            return TimeAwareCalibratedGBM(
                n_estimators=hp["n_estimators"],
                learning_rate=hp["learning_rate"],
                max_depth=hp["max_depth"],
                random_state=hp["random_state"],
                calibration_fraction=hp["calibration_fraction"],
                min_calibration_rows=hp["min_calibration_rows"],
            )
        return GradientBoostingClassifier(
            n_estimators=hp["n_estimators"],
            learning_rate=hp["learning_rate"],
            max_depth=hp["max_depth"],
            random_state=hp["random_state"],
        )

    return factory


def walk_forward_window(
    df: pd.DataFrame,
    features: list,
    target: str,
    window_start: datetime,
    window_end: datetime,
    estimator_factory: Callable,
) -> pd.DataFrame | None:
    """Walk-forward over a single date window.

    Trains on all games in `df` strictly before each weekly cutoff; tests on
    home-team rows within that week (home-only to avoid double-counting each
    game). Returns a per-prediction DataFrame with prob_home, target, conf.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Config references features not in frozen CSV: {missing}")

    fold_logs = []
    current = window_start
    while current < window_end:
        next_week = current + timedelta(days=7)

        train = df[df["date"] < current].dropna(subset=features + [target])
        if len(train) < MIN_TRAIN_ROWS:
            current = next_week
            continue

        week_mask = (
            (df["date"] >= current)
            & (df["date"] < next_week)
            & (df["is_home"] == 1)
        )
        week = df.loc[week_mask].dropna(subset=features + [target])
        if week.empty:
            current = next_week
            continue

        est = estimator_factory()
        est.fit(train[features].astype(float), train[target].astype(int))
        probs = est.predict_proba(week[features].astype(float))[:, 1]

        fold_logs.append(
            pd.DataFrame(
                {
                    "date": week["date"].values,
                    "prob_home": probs,
                    "target": week[target].astype(int).values,
                    "conf": np.maximum(probs, 1 - probs),
                }
            )
        )
        current = next_week

    if not fold_logs:
        return None
    return pd.concat(fold_logs, ignore_index=True)


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
    # Clip for numerical stability. sklearn log_loss handles this internally
    # with epsilon but being explicit keeps the metric reproducible.
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    ll = float(log_loss(y, p_clipped, labels=[0, 1]))
    acc = float((pred_class == y).mean())

    hc_mask = predictions["conf"] >= high_conf_threshold
    n_hc = int(hc_mask.sum())
    if n_hc:
        hc_correct = int((pred_class[hc_mask.values] == y[hc_mask.values]).sum())
        hc_acc = hc_correct / n_hc
        payout = 100.0 / 110.0  # -110 break-even payout
        roi = (hc_correct * payout) - (n_hc - hc_correct)
    else:
        hc_acc = None
        roi = 0.0

    return {
        "brier": brier,
        "log_loss": ll,
        "accuracy": acc,
        "high_conf_accuracy": hc_acc,
        "roi_units": float(roi),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to model config JSON. Defaults to production MLB setup.",
    )
    args = parser.parse_args()

    if args.model_config:
        with open(args.model_config) as f:
            config = json.load(f)
    else:
        config = {}

    features = config.get("features") or DEFAULT_MLB_FEATURES
    target = config.get("target", "home_win")
    high_conf_threshold = config.get("high_conf_threshold", HIGH_CONF_THRESHOLD_DEFAULT)

    manifest = load_manifest()
    df = load_frozen_df()
    factory = build_estimator_factory(config)

    results = {}
    for key in ("optimizer", "monitor_2025_tail", "monitor_2026"):
        start, end_exclusive = parse_window_bounds(manifest, key)
        preds = walk_forward_window(df, features, target, start, end_exclusive, factory)
        results[key] = summarize(preds, high_conf_threshold)

    results["_meta"] = {
        "features_used": features,
        "n_features": len(features),
        "calibrated": config.get("calibrated", True),
        "target": target,
        "high_conf_threshold": high_conf_threshold,
        "anchor_sha256": manifest["sha256"],
        "hyperparams": {**DEFAULT_HYPERPARAMS, **(config.get("hyperparams") or {})},
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
