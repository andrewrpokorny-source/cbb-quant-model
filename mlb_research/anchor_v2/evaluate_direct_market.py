"""Evaluate the frozen market no-vig probability as a direct baseline."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from mlb_research.anchor.anchor_eval import (
    HIGH_CONF_THRESHOLD,
    MIN_TRAIN_ROWS,
    atomic_write_json,
    load_frozen_df,
    load_manifest,
    parse_window_bounds,
    summarize,
)

PROB_COLUMN = "market_home_no_vig_prob"
REQUIRED_COLUMNS = [
    PROB_COLUMN,
    "home_win",
    "team_moneyline",
    "opp_moneyline",
]


def direct_market_predictions(
    df: pd.DataFrame,
    window_start,
    window_end,
    extra_required_columns: list[str] | None = None,
) -> tuple[pd.DataFrame | None, dict]:
    """Mirror anchor_eval's weekly window shape without fitting a model."""
    dropna_columns = list(dict.fromkeys(REQUIRED_COLUMNS + (extra_required_columns or [])))
    missing = [column for column in dropna_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Frozen CSV is missing market baseline columns: {missing}")

    fold_logs = []
    week_counts = []
    skipped_thin_train = 0
    skipped_empty_week = 0

    current = window_start
    while current < window_end:
        next_week = current + timedelta(days=7)
        train = df[df["date"] < current].dropna(subset=dropna_columns)
        if len(train) < MIN_TRAIN_ROWS:
            skipped_thin_train += 1
            current = next_week
            continue

        week_mask = (
            (df["date"] >= current)
            & (df["date"] < next_week)
            & (df["is_home"] == 1)
        )
        week = df.loc[week_mask].dropna(subset=dropna_columns)
        if week.empty:
            skipped_empty_week += 1
            current = next_week
            continue

        probs = week[PROB_COLUMN].astype(float)
        fold_logs.append(
            pd.DataFrame(
                {
                    "date": week["date"].values,
                    "prob_home": probs.values,
                    "target": week["home_win"].astype(int).values,
                    "conf": np.maximum(probs, 1 - probs),
                    "home_moneyline": week["team_moneyline"].values,
                    "away_moneyline": week["opp_moneyline"].values,
                }
            )
        )
        week_counts.append(int(len(week)))
        current = next_week

    diagnostics = {
        "n_folds_trained": len(fold_logs),
        "n_folds_skipped_thin_train": skipped_thin_train,
        "n_folds_skipped_empty_week": skipped_empty_week,
        "week_rows_min": min(week_counts) if week_counts else None,
        "week_rows_max": max(week_counts) if week_counts else None,
        "week_rows_mean": (sum(week_counts) / len(week_counts)) if week_counts else None,
    }
    if not fold_logs:
        return None, diagnostics
    return pd.concat(fold_logs, ignore_index=True), diagnostics


def config_dropna_columns(model_config_path: str | None) -> tuple[list[str], dict | None]:
    if not model_config_path:
        return [], None
    with open(model_config_path) as f:
        config = json.load(f)
    features = config.get("features") or []
    target = config.get("target", "home_win")
    return list(dict.fromkeys([*features, target, "home_win"])), config


def evaluate_direct_market(model_config_path: str | None = None) -> dict:
    manifest = load_manifest()
    df = load_frozen_df()
    dropna_columns, config = config_dropna_columns(model_config_path)

    results = {}
    diagnostics_by_window = {}
    for key in ("optimizer", "monitor_2025_tail", "monitor_2026"):
        start, end_exclusive = parse_window_bounds(manifest, key)
        preds, diag = direct_market_predictions(
            df,
            start,
            end_exclusive,
            extra_required_columns=dropna_columns,
        )
        results[key] = summarize(
            preds,
            HIGH_CONF_THRESHOLD,
            roi_mode="moneyline",
        )
        diagnostics_by_window[key] = diag

    results["_meta"] = {
        "baseline": "direct_market_no_vig",
        "probability_column": PROB_COLUMN,
        "roi_mode": "moneyline",
        "high_conf_threshold": HIGH_CONF_THRESHOLD,
        "anchor_sha256": manifest["sha256"],
        "row_mask_config_path": model_config_path,
        "row_mask_n_features": len(config.get("features", [])) if config else None,
        "diagnostics": diagnostics_by_window,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default=None,
        help=(
            "Optional config whose feature/target columns define the row mask. "
            "Use this to compare market-only probabilities on the same rows as "
            "a model candidate."
        ),
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    results = evaluate_direct_market(args.model_config)
    if args.output:
        atomic_write_json(args.output, results)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
