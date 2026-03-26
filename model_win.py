"""Train calibrated game-winner (P(home wins)) models for CBB."""

import argparse
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss

from league_config import get_league_artifact_paths, normalize_league
from model import TimeAwareCalibratedGBM


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Works even when no market spread is available.
WOMENS_FEATURES_NO_LINE = [
    "is_neutral",
    "rest_days",
    "prev_games_played",
    "opp_win_pct",
    "prev_blowout_rate",
    "prev_roll5_margin",
    "prev_volatility",
    "prev_win_pct",
    "distance_advantage",
    "prev_season_team_score",
    "prev_roll3_team_score",
]
MENS_FEATURES_NO_LINE = [
    "is_neutral",
    "rest_days",
    "roll5_cover_margin",
    "prev_games_played",
    "opp_win_pct",
    "prev_blowout_rate",
    "prev_roll5_margin",
    "prev_volatility",
]
WIN_FEATURES_NO_LINE_BY_LEAGUE = {
    "mens": MENS_FEATURES_NO_LINE,
    "womens": WOMENS_FEATURES_NO_LINE,
}

# Adds market context where available.
WOMENS_FEATURES_WITH_LINE = WOMENS_FEATURES_NO_LINE + [
    "spread",
    "spread_abs",
    "spread_squared",
]
MENS_FEATURES_WITH_LINE = MENS_FEATURES_NO_LINE + [
    "spread",
    "spread_abs",
    "spread_squared",
]
WIN_FEATURES_WITH_LINE_BY_LEAGUE = {
    "mens": MENS_FEATURES_WITH_LINE,
    "womens": WOMENS_FEATURES_WITH_LINE,
}


DEFAULT_FILL = {
    "is_neutral": 0.0,
    "rest_days": 3.0,
    "diff_eFG": 0.0,
    "diff_Rebound": 0.0,
    "diff_TO": 0.0,
    "momentum_gap": 0.0,
    "roll5_cover_margin": 0.0,
    "prev_games_played": 10.0,
    "opp_win_pct": 0.5,
    "prev_blowout_rate": 0.0,
    "prev_roll5_margin": 0.0,
    "prev_volatility": 10.0,
    "prev_win_pct": 0.5,
    "distance_advantage": 0.0,
    "prev_season_team_score": 70.0,
    "prev_roll3_team_score": 70.0,
    "diff_prev_season_team_score": 0.0,
    "diff_prev_roll3_team_score": 0.0,
    "prev_season_off_rating": 100.0,
    "opp_season_off_rating": 100.0,
    "off_rating_gap": 0.0,
    "spread": 0.0,
    "spread_abs": 0.0,
    "spread_squared": 0.0,
}


def _base_estimator():
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )


def _prepare_home_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare one row per game from the home-team perspective."""
    frame = df.copy()

    if "is_home" in frame.columns:
        frame = frame[frame["is_home"] == 1].copy()
    elif "location" in frame.columns:
        frame = frame[frame["location"].astype(str).str.lower() == "home"].copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    frame = frame.sort_values("date")

    frame["home_win"] = (frame["team_score"] > frame["opp_score"]).astype(int)
    if "spread" not in frame.columns:
        frame["spread"] = 0.0
    frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce").fillna(0.0)
    frame["spread_abs"] = frame["spread"].abs()
    frame["spread_squared"] = frame["spread"] ** 2

    # Derived quality differential.
    frame["off_rating_gap"] = (
        _numeric_series(frame, "prev_season_off_rating", 100.0)
        - _numeric_series(frame, "opp_season_off_rating", 100.0)
    )

    return frame


def _build_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    for col in features:
        if col not in frame.columns:
            frame[col] = DEFAULT_FILL.get(col, 0.0)
    X = frame[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.fillna({k: DEFAULT_FILL[k] for k in features}).fillna(0.0)


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Return a numeric series for a column or a default-filled fallback."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _train_variant(frame: pd.DataFrame, features: list[str], name: str):
    """Train uncalibrated + calibrated model variant."""
    df_model = frame.copy()
    X = _build_features(df_model, features)
    y = df_model["home_win"].astype(int)

    if len(X) < 200:
        print(f"   Skipping {name}: not enough samples ({len(X)}).")
        return None
    if y.nunique() < 2:
        print(f"   Skipping {name}: only one target class.")
        return None

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_test) == 0 or y_train.nunique() < 2:
        print(f"   Skipping {name}: invalid train/test split.")
        return None

    uncal = _base_estimator()
    uncal.fit(X_train, y_train)
    uncal_probs = uncal.predict_proba(X_test)[:, 1]
    uncal_preds = (uncal_probs >= 0.5).astype(int)

    cal = TimeAwareCalibratedGBM(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    cal.fit(X_train, y_train)
    cal_probs = cal.predict_proba(X_test)[:, 1]
    cal_preds = (cal_probs >= 0.5).astype(int)

    metrics = {
        "variant": name,
        "samples": int(len(X)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "uncal_accuracy": float(accuracy_score(y_test, uncal_preds)),
        "uncal_brier": float(brier_score_loss(y_test, uncal_probs)),
        "cal_accuracy": float(accuracy_score(y_test, cal_preds)),
        "cal_brier": float(brier_score_loss(y_test, cal_probs)),
    }
    return {
        "model": cal,
        "features": features,
        "metrics": metrics,
    }


def train_win_models(league: str = "mens"):
    """Train no-line and with-line P(home wins) models."""
    league = normalize_league(league)
    paths = get_league_artifact_paths(BASE_DIR, league)
    data_file = paths["data_file"]
    out_file = paths["win_model_file"]

    print(f"--- TRAINING P(WIN) MODELS ({league}) ---")
    if not os.path.exists(data_file):
        print(f"No processed data found at {data_file}. Run features.py first.")
        return None

    df = pd.read_csv(data_file, low_memory=False)
    print(f"Loaded {len(df)} rows from {os.path.basename(data_file)}")
    home_rows = _prepare_home_rows(df)
    print(f"Home-game rows available: {len(home_rows)}")

    no_line_features = get_win_feature_list(league, with_line=False)
    with_line_features = get_win_feature_list(league, with_line=True)

    no_line = _train_variant(home_rows, no_line_features, "no_line")

    with_line_rows = home_rows[home_rows["spread"].abs() > 0].copy()
    print(f"Rows with non-zero spread: {len(with_line_rows)}")
    with_line = _train_variant(with_line_rows, with_line_features, "with_line")

    if no_line is None and with_line is None:
        print("No P(win) variant could be trained.")
        return None

    payload = {
        "league": league,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_no_line": no_line["model"] if no_line else None,
        "features_no_line": no_line["features"] if no_line else no_line_features,
        "metrics_no_line": no_line["metrics"] if no_line else None,
        "model_with_line": with_line["model"] if with_line else None,
        "features_with_line": with_line["features"] if with_line else with_line_features,
        "metrics_with_line": with_line["metrics"] if with_line else None,
    }
    joblib.dump(payload, out_file)
    print(f"Saved P(win) bundle to {out_file}")

    for key in ("metrics_no_line", "metrics_with_line"):
        m = payload.get(key)
        if not m:
            continue
        print(
            f"   {m['variant']}: "
            f"acc={m['cal_accuracy']:.1%}, "
            f"brier={m['cal_brier']:.4f}, "
            f"n={m['samples']}"
        )
    return payload


def load_win_model_bundle(path: str | None = None, league: str = "mens") -> dict:
    """Load a trained P(win) bundle."""
    if path is None:
        league = normalize_league(league)
        path = get_league_artifact_paths(BASE_DIR, league)["win_model_file"]
    data = joblib.load(path)
    if isinstance(data, dict) and ("model_no_line" in data or "model_with_line" in data):
        return data
    # Backward compatibility: raw model treated as no-line variant.
    return {
        "model_no_line": data,
        "features_no_line": get_win_feature_list(league, with_line=False),
        "metrics_no_line": None,
        "model_with_line": None,
        "features_with_line": get_win_feature_list(league, with_line=True),
        "metrics_with_line": None,
    }


def get_win_feature_list(league: str = "mens", *, with_line: bool = False) -> list[str]:
    """Return the league-specific P(win) feature list."""
    canonical = normalize_league(league)
    mapping = WIN_FEATURES_WITH_LINE_BY_LEAGUE if with_line else WIN_FEATURES_NO_LINE_BY_LEAGUE
    return list(mapping[canonical])


def predict_home_win_prob(
    row: dict | pd.Series,
    bundle: dict,
    allow_with_line: bool = True,
) -> tuple[float, str]:
    """
    Predict P(home wins) from a feature row and model bundle.

    Returns:
        (probability, variant_used)
    """
    if isinstance(row, pd.Series):
        row_data = row.to_dict()
    else:
        row_data = dict(row)

    has_spread = abs(float(row_data.get("spread", 0) or 0)) > 0
    use_with_line = (
        allow_with_line
        and has_spread
        and bundle.get("model_with_line") is not None
    )

    if use_with_line:
        model = bundle["model_with_line"]
        features = bundle.get("features_with_line")
        if not features:
            raise ValueError("Win model bundle missing features_with_line.")
        variant = "with_line"
    else:
        model = bundle.get("model_no_line")
        if model is None:
            raise ValueError("No available no-line model in bundle.")
        features = bundle.get("features_no_line")
        if not features:
            raise ValueError("Win model bundle missing features_no_line.")
        variant = "no_line"

    frame = pd.DataFrame([row_data])
    X = _build_features(frame, features)
    prob_home = float(model.predict_proba(X)[0][1])
    return prob_home, variant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CBB game-winner P(win) model bundle.")
    parser.add_argument(
        "--league",
        default="mens",
        help="League to train: mens or womens (aliases supported).",
    )
    args = parser.parse_args()
    train_win_models(args.league)
