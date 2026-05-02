"""MLB market-v2 shadow model helpers.

The market-v2 model is production-shadow only: it emits comparison columns for
daily MLB predictions but does not drive picks, ratings, or staking.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from mlb_research.market_odds import (  # noqa: E402
    american_odds_to_implied_prob,
    no_vig_probability,
)

MARKET_V2_MODEL_FILE = os.path.join(BASE_DIR, "models", "mlb_market_v2_shadow_model.pkl")
MARKET_V2_TRAINING_FILE = os.path.join(
    BASE_DIR,
    "mlb_research",
    "anchor_v2",
    "mlb_market_frozen.csv",
)

MARKET_V2_FEATURES = [
    "bullpen_era_diff",
    "sp_era_diff",
    "is_home",
    "opp_sp_roll_era",
    "sp_roll_era",
    "sp_roll_era_diff",
    "sp_era",
    "sp_roll_ip",
    "prev_roll10_runs_allowed",
    "prev_season_runs_per_game",
    "opp_sp_era",
    "roll5_rpg_diff",
    "roll10_ra_diff",
    "market_home_no_vig_prob",
]

MODEL_HYPERPARAMS = {
    "n_estimators": 150,
    "max_depth": 1,
    "learning_rate": 0.05,
    "random_state": 42,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def market_v2_shadow_enabled() -> bool:
    return os.environ.get("MLB_MARKET_V2_SHADOW", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def market_home_no_vig_probability(home_moneyline: Any, away_moneyline: Any) -> float:
    home = american_odds_to_implied_prob(home_moneyline)
    away = american_odds_to_implied_prob(away_moneyline)
    return no_vig_probability(home, away)


def _finite_probability(value: Any) -> bool:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(prob) and 0.0 <= prob <= 1.0


def train_market_v2_model(
    source_csv: str = MARKET_V2_TRAINING_FILE,
    output_path: str = MARKET_V2_MODEL_FILE,
) -> dict:
    """Train and persist the market-v2 shadow model artifact."""
    from lightgbm import LGBMClassifier

    df = pd.read_csv(source_csv, low_memory=False)
    required = [*MARKET_V2_FEATURES, "home_win"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Market-v2 training CSV missing required columns: {missing}")

    df_model = df.dropna(subset=required).copy()
    if df_model.empty:
        raise ValueError("Market-v2 training frame is empty after dropna.")

    model = LGBMClassifier(**MODEL_HYPERPARAMS, verbosity=-1)
    model.fit(
        df_model[MARKET_V2_FEATURES].astype(float),
        df_model["home_win"].astype(int),
    )

    bundle = {
        "model": model,
        "features": list(MARKET_V2_FEATURES),
        "model_family": "lightgbm",
        "hyperparams": dict(MODEL_HYPERPARAMS),
        "target": "home_win",
        "source_csv": os.path.relpath(source_csv, BASE_DIR),
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "training_rows": int(len(df_model)),
        "shadow_only": True,
        "description": "MLB market-v2 LightGBM top-13 + market feature shadow model",
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    return bundle


def load_market_v2_model(path: str = MARKET_V2_MODEL_FILE) -> dict | None:
    if not market_v2_shadow_enabled():
        return None
    if not os.path.exists(path):
        return None
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid market-v2 model bundle at {path}")
    if not bundle.get("features"):
        raise ValueError(f"Market-v2 model bundle at {path} missing features.")
    return bundle


def predict_market_v2_home_prob(
    bundle: dict,
    base_feature_row: dict,
    home_moneyline: Any,
    away_moneyline: Any,
) -> dict:
    """Return market-v2 shadow prediction metadata for one MLB game."""
    market_home_prob = market_home_no_vig_probability(home_moneyline, away_moneyline)
    if not _finite_probability(market_home_prob):
        return {
            "status": "odds_missing",
            "market_home_no_vig_prob": None,
        }

    features = list(bundle["features"])
    row = dict(base_feature_row)
    row["market_home_no_vig_prob"] = market_home_prob
    missing = [feature for feature in features if feature not in row]
    if missing:
        return {
            "status": "feature_missing",
            "missing_features": missing,
            "market_home_no_vig_prob": market_home_prob,
        }

    X = pd.DataFrame([row])[features].astype(float)
    prob_home = float(bundle["model"].predict_proba(X)[:, 1][0])
    if not _finite_probability(prob_home):
        return {
            "status": "invalid_prediction",
            "market_home_no_vig_prob": market_home_prob,
        }

    pick_home = prob_home > 0.5
    selected_model_prob = prob_home if pick_home else 1.0 - prob_home
    selected_market_prob = market_home_prob if pick_home else 1.0 - market_home_prob
    return {
        "status": "ok",
        "prob_home": prob_home,
        "prob_away": 1.0 - prob_home,
        "conf": selected_model_prob,
        "pick_home": pick_home,
        "market_home_no_vig_prob": market_home_prob,
        "edge_vs_market": selected_model_prob - selected_market_prob,
    }


def empty_shadow_columns(status: str) -> dict:
    return {
        "MarketV2_Status": status,
        "MarketV2_Prob_Home": None,
        "MarketV2_Prob_Away": None,
        "MarketV2_Pick": "",
        "MarketV2_Conf": None,
        "MarketV2_Market_NoVig_Home": None,
        "MarketV2_Edge_vs_Market": None,
        "MarketV2_Agrees_With_Production": "",
    }


def build_shadow_columns(
    bundle: dict | None,
    base_feature_row: dict,
    home_team: str,
    away_team: str,
    home_moneyline: Any,
    away_moneyline: Any,
    production_pick: str,
    unavailable_status: str = "model_missing",
) -> dict:
    if bundle is None:
        return empty_shadow_columns(unavailable_status)

    result = predict_market_v2_home_prob(
        bundle,
        base_feature_row,
        home_moneyline,
        away_moneyline,
    )
    columns = empty_shadow_columns(result["status"])
    market_prob = result.get("market_home_no_vig_prob")
    columns["MarketV2_Market_NoVig_Home"] = (
        round(float(market_prob), 4) if _finite_probability(market_prob) else None
    )
    if result["status"] != "ok":
        return columns

    pick = home_team if result["pick_home"] else away_team
    columns.update(
        {
            "MarketV2_Prob_Home": round(float(result["prob_home"]), 3),
            "MarketV2_Prob_Away": round(float(result["prob_away"]), 3),
            "MarketV2_Pick": pick,
            "MarketV2_Conf": round(float(result["conf"]), 3),
            "MarketV2_Edge_vs_Market": round(float(result["edge_vs_market"]), 4),
            "MarketV2_Agrees_With_Production": bool(pick == production_pick),
        }
    )
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLB market-v2 shadow model.")
    parser.add_argument("--source", default=MARKET_V2_TRAINING_FILE)
    parser.add_argument("--output", default=MARKET_V2_MODEL_FILE)
    args = parser.parse_args()

    bundle = train_market_v2_model(args.source, args.output)
    print(
        f"Saved market-v2 shadow model to {args.output} "
        f"({bundle['training_rows']} rows)"
    )


if __name__ == "__main__":
    main()
