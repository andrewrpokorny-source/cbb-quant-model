"""Paired diagnostics for a market-aware MLB model candidate."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault(
    "MLB_RESEARCH_FROZEN_CSV",
    "mlb_research/anchor_v2/mlb_market_frozen.csv",
)
os.environ.setdefault(
    "MLB_RESEARCH_ANCHOR_MANIFEST",
    "mlb_research/anchor_v2/market_anchor_manifest.json",
)

from mlb_research.anchor.anchor_eval import (  # noqa: E402
    DEFAULT_MLB_FEATURES,
    HIGH_CONF_THRESHOLD,
    atomic_write_json,
    build_estimator_factory,
    load_frozen_df,
    load_manifest,
    parse_window_bounds,
    summarize,
    validate_config_keys,
    walk_forward_window,
)
from mlb_research.anchor_v2.evaluate_direct_market import (  # noqa: E402
    direct_market_predictions,
)
from mlb_research.market_odds import american_odds_profit  # noqa: E402

WINDOW_KEYS = ("optimizer", "monitor_2025_tail", "monitor_2026")
EPSILON = 1e-6


def log_loss_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p.astype(float), EPSILON, 1 - EPSILON)
    return -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))


def side_roi_units(
    prob_home: float,
    target: int,
    home_moneyline: float,
    away_moneyline: float,
    threshold: float = HIGH_CONF_THRESHOLD,
) -> tuple[float, bool]:
    conf = max(prob_home, 1 - prob_home)
    if conf < threshold:
        return 0.0, False
    pick_home = prob_home > 0.5
    selected_odds = home_moneyline if pick_home else away_moneyline
    profit = american_odds_profit(selected_odds)
    if not math.isfinite(profit):
        return 0.0, False
    won = target == int(pick_home)
    return (profit if won else -1.0), True


def paired_prediction_frame(
    model_preds: pd.DataFrame,
    market_preds: pd.DataFrame,
) -> pd.DataFrame:
    if len(model_preds) != len(market_preds):
        raise ValueError(
            "Model and market prediction frames have different lengths: "
            f"{len(model_preds)} vs {len(market_preds)}."
        )
    for column in ("date", "target", "home_moneyline", "away_moneyline"):
        left = model_preds[column].reset_index(drop=True)
        right = market_preds[column].reset_index(drop=True)
        if column == "date":
            left = pd.to_datetime(left)
            right = pd.to_datetime(right)
        if not left.equals(right):
            raise ValueError(f"Model and market prediction frames differ on {column!r}.")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(model_preds["date"]).reset_index(drop=True),
            "target": model_preds["target"].astype(int).reset_index(drop=True),
            "model_prob_home": model_preds["prob_home"].astype(float).reset_index(drop=True),
            "market_prob_home": market_preds["prob_home"].astype(float).reset_index(drop=True),
            "home_moneyline": model_preds["home_moneyline"].reset_index(drop=True),
            "away_moneyline": model_preds["away_moneyline"].reset_index(drop=True),
        }
    )
    frame["model_conf"] = np.maximum(frame["model_prob_home"], 1 - frame["model_prob_home"])
    frame["market_conf"] = np.maximum(
        frame["market_prob_home"], 1 - frame["market_prob_home"]
    )
    frame["model_pick_home"] = frame["model_prob_home"] > 0.5
    frame["market_pick_home"] = frame["market_prob_home"] > 0.5
    frame["model_selected_market_prob"] = np.where(
        frame["model_pick_home"],
        frame["market_prob_home"],
        1 - frame["market_prob_home"],
    )
    frame["model_selected_side_type"] = np.where(
        frame["model_selected_market_prob"] >= 0.5,
        "market_favorite",
        "market_dog",
    )
    return frame


def prediction_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n_games": 0,
            "model_brier": None,
            "market_brier": None,
            "brier_delta": None,
            "model_log_loss": None,
            "market_log_loss": None,
            "log_loss_delta": None,
            "model_roi_units": None,
            "market_roi_units": None,
            "roi_units_delta": None,
            "model_n_hc": 0,
            "market_n_hc": 0,
            "model_hc_accuracy": None,
            "market_hc_accuracy": None,
            "model_market_pick_agreement": None,
        }

    y = frame["target"].astype(int).to_numpy()
    model_p = frame["model_prob_home"].astype(float).to_numpy()
    market_p = frame["market_prob_home"].astype(float).to_numpy()
    model_brier_loss = (model_p - y) ** 2
    market_brier_loss = (market_p - y) ** 2
    model_log_loss = log_loss_vector(y, model_p)
    market_log_loss = log_loss_vector(y, market_p)

    model_roi = []
    market_roi = []
    model_hc = []
    market_hc = []
    for row in frame.itertuples(index=False):
        units, priced = side_roi_units(
            row.model_prob_home,
            row.target,
            row.home_moneyline,
            row.away_moneyline,
        )
        model_roi.append(units)
        model_hc.append(priced)
        units, priced = side_roi_units(
            row.market_prob_home,
            row.target,
            row.home_moneyline,
            row.away_moneyline,
        )
        market_roi.append(units)
        market_hc.append(priced)

    model_hc_mask = np.array(model_hc, dtype=bool)
    market_hc_mask = np.array(market_hc, dtype=bool)
    model_pick = model_p > 0.5
    market_pick = market_p > 0.5

    def high_conf_accuracy(pick: np.ndarray, mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        return float((pick[mask].astype(int) == y[mask]).mean())

    return {
        "n_games": int(len(frame)),
        "model_brier": float(model_brier_loss.mean()),
        "market_brier": float(market_brier_loss.mean()),
        "brier_delta": float(model_brier_loss.mean() - market_brier_loss.mean()),
        "model_log_loss": float(model_log_loss.mean()),
        "market_log_loss": float(market_log_loss.mean()),
        "log_loss_delta": float(model_log_loss.mean() - market_log_loss.mean()),
        "model_roi_units": float(np.sum(model_roi)),
        "market_roi_units": float(np.sum(market_roi)),
        "roi_units_delta": float(np.sum(model_roi) - np.sum(market_roi)),
        "model_n_hc": int(model_hc_mask.sum()),
        "market_n_hc": int(market_hc_mask.sum()),
        "model_hc_accuracy": high_conf_accuracy(model_pick, model_hc_mask),
        "market_hc_accuracy": high_conf_accuracy(market_pick, market_hc_mask),
        "model_market_pick_agreement": float((model_pick == market_pick).mean()),
    }


def bootstrap_deltas(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict:
    if frame.empty or samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(frame)
    y = frame["target"].astype(int).to_numpy()
    model_p = frame["model_prob_home"].astype(float).to_numpy()
    market_p = frame["market_prob_home"].astype(float).to_numpy()
    brier_delta = (model_p - y) ** 2 - (market_p - y) ** 2
    log_loss_delta = log_loss_vector(y, model_p) - log_loss_vector(y, market_p)

    model_roi = []
    market_roi = []
    for row in frame.itertuples(index=False):
        model_units, _ = side_roi_units(
            row.model_prob_home,
            row.target,
            row.home_moneyline,
            row.away_moneyline,
        )
        market_units, _ = side_roi_units(
            row.market_prob_home,
            row.target,
            row.home_moneyline,
            row.away_moneyline,
        )
        model_roi.append(model_units)
        market_roi.append(market_units)
    roi_delta = np.array(model_roi) - np.array(market_roi)

    brier_samples = []
    log_loss_samples = []
    roi_samples = []
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        brier_samples.append(float(brier_delta[idx].mean()))
        log_loss_samples.append(float(log_loss_delta[idx].mean()))
        roi_samples.append(float(roi_delta[idx].sum()))

    def interval(values: list[float]) -> dict:
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "p025": float(np.quantile(arr, 0.025)),
            "p975": float(np.quantile(arr, 0.975)),
        }

    return {
        "samples": samples,
        "seed": seed,
        "brier_delta": interval(brier_samples),
        "log_loss_delta": interval(log_loss_samples),
        "roi_units_delta": interval(roi_samples),
    }


def grouped_summaries(frame: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    monthly = {}
    for month, group in frame.groupby(frame["date"].dt.to_period("M").astype(str)):
        monthly[month] = prediction_summary(group)
    out["monthly"] = monthly

    side_type = {}
    for label, group in frame.groupby("model_selected_side_type"):
        side_type[label] = prediction_summary(group)
    out["model_selected_side_type"] = side_type

    confidence_decile = {}
    if len(frame) >= 10 and frame["model_conf"].nunique() > 1:
        deciles = pd.qcut(
            frame["model_conf"],
            q=min(10, len(frame)),
            labels=False,
            duplicates="drop",
        )
        for decile, group in frame.groupby(deciles):
            confidence_decile[str(int(decile))] = prediction_summary(group)
    out["model_confidence_decile"] = confidence_decile
    return out


def run_diagnostics(
    model_config_path: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    with open(model_config_path) as f:
        config = json.load(f)
    validate_config_keys(config)

    manifest = load_manifest()
    df = load_frozen_df()
    features = config.get("features") or DEFAULT_MLB_FEATURES
    target = config.get("target", "home_win")
    factory = build_estimator_factory(config)
    direct_dropna_cols = list(dict.fromkeys([*features, target, "home_win"]))

    windows = {}
    for offset, key in enumerate(WINDOW_KEYS):
        start, end_exclusive = parse_window_bounds(manifest, key)
        model_preds, model_diag = walk_forward_window(
            df,
            features,
            target,
            start,
            end_exclusive,
            factory,
        )
        market_preds, market_diag = direct_market_predictions(
            df,
            start,
            end_exclusive,
            extra_required_columns=direct_dropna_cols,
        )
        if model_preds is None or market_preds is None:
            paired = pd.DataFrame()
        else:
            paired = paired_prediction_frame(model_preds, market_preds)
        windows[key] = {
            "summary": prediction_summary(paired),
            "bootstrap": bootstrap_deltas(
                paired,
                samples=bootstrap_samples,
                seed=seed + offset,
            ),
            "groups": grouped_summaries(paired),
            "model_anchor_eval_summary": summarize(
                model_preds,
                HIGH_CONF_THRESHOLD,
                roi_mode=config.get("roi_mode", "moneyline"),
            ),
            "market_anchor_eval_summary": summarize(
                market_preds,
                HIGH_CONF_THRESHOLD,
                roi_mode="moneyline",
            ),
            "model_diagnostics": model_diag,
            "market_diagnostics": market_diag,
        }

    return {
        "_meta": {
            "model_config_path": model_config_path,
            "anchor_sha256": manifest["sha256"],
            "high_conf_threshold": HIGH_CONF_THRESHOLD,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "interpretation": "negative brier/log_loss deltas favor the model over market-only",
        },
        "windows": windows,
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Market V2 Production-Readiness Diagnostics",
        "",
        f"Config: `{results['_meta']['model_config_path']}`",
        f"Anchor sha256: `{results['_meta']['anchor_sha256']}`",
        "",
        "| Window | N | Brier Delta | Log Loss Delta | ROI Delta | Model ROI | Market ROI | Pick Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in WINDOW_KEYS:
        s = results["windows"][key]["summary"]
        lines.append(
            "| {key} | {n} | {brier:.6f} | {ll:.6f} | {roi:.2f} | "
            "{model_roi:.2f} | {market_roi:.2f} | {agree:.3f} |".format(
                key=key,
                n=s["n_games"],
                brier=s["brier_delta"],
                ll=s["log_loss_delta"],
                roi=s["roi_units_delta"],
                model_roi=s["model_roi_units"],
                market_roi=s["market_roi_units"],
                agree=s["model_market_pick_agreement"],
            )
        )

    lines.extend(["", "## Bootstrap 95% Intervals", ""])
    for key in WINDOW_KEYS:
        b = results["windows"][key]["bootstrap"]
        lines.append(
            "- `{}`: Brier delta [{:.6f}, {:.6f}], log loss delta "
            "[{:.6f}, {:.6f}], ROI delta [{:.2f}, {:.2f}]".format(
                key,
                b["brier_delta"]["p025"],
                b["brier_delta"]["p975"],
                b["log_loss_delta"]["p025"],
                b["log_loss_delta"]["p975"],
                b["roi_units_delta"]["p025"],
                b["roi_units_delta"]["p975"],
            )
        )

    lines.extend(["", "## Production Read", ""])
    optimizer = results["windows"]["optimizer"]["summary"]
    mon25 = results["windows"]["monitor_2025_tail"]["summary"]
    mon26 = results["windows"]["monitor_2026"]["summary"]
    all_brier_better = all(
        results["windows"][key]["summary"]["brier_delta"] < 0 for key in WINDOW_KEYS
    )
    monitors_roi_positive = mon25["model_roi_units"] > 0 and mon26["model_roi_units"] > 0
    if all_brier_better and monitors_roi_positive:
        lines.append(
            "The candidate clears the first production-readiness screen: paired "
            "Brier is better than market-only in every frozen window and monitor "
            "ROI remains positive. Next step is prediction-time odds plumbing, "
            "with this model still behind a comparison/reporting flag."
        )
    else:
        lines.append(
            "The candidate does not clear the first production-readiness screen. "
            "Investigate the failing split before production wiring."
        )
    lines.append(
        "Optimizer ROI should not be treated as a promotion criterion by itself; "
        f"the paired optimizer Brier delta is `{optimizer['brier_delta']:.6f}`."
    )
    lines.extend(["", "## Watchouts", ""])
    for key in WINDOW_KEYS:
        b = results["windows"][key]["bootstrap"]["brier_delta"]
        if b["p025"] <= 0 <= b["p975"]:
            lines.append(
                f"- `{key}` paired Brier bootstrap interval crosses zero: "
                f"[{b['p025']:.6f}, {b['p975']:.6f}]."
            )
    monthly_flags = []
    for key in WINDOW_KEYS:
        monthly = results["windows"][key]["groups"]["monthly"]
        for month, summary in monthly.items():
            if summary["n_games"] >= 25 and summary["brier_delta"] > 0:
                monthly_flags.append(
                    f"`{key}` {month} n={summary['n_games']} "
                    f"Brier delta={summary['brier_delta']:.6f}"
                )
    if monthly_flags:
        lines.append(
            "- Some month-level splits are worse than market-only: "
            + "; ".join(monthly_flags)
            + "."
        )
    side_groups = results["windows"]["optimizer"]["groups"]["model_selected_side_type"]
    if {"market_dog", "market_favorite"}.issubset(side_groups):
        dog_roi = side_groups["market_dog"]["roi_units_delta"]
        favorite_roi = side_groups["market_favorite"]["roi_units_delta"]
        lines.append(
            "- Optimizer ROI edge is concentrated in model-selected dogs: "
            f"dog ROI delta `{dog_roi:.2f}U`, favorite ROI delta "
            f"`{favorite_roi:.2f}U`."
        )
    if mon26["n_games"] < 100:
        lines.append(
            f"- Monitor 2026 is still a small sample (`n={mon26['n_games']}`); "
            "use it as a smoke check, not a promotion proof."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        default="mlb_research/anchor_v2/configs/lgbm_top13_market.json",
        help="Model config to compare against the direct market no-vig baseline.",
    )
    parser.add_argument(
        "--output",
        default="mlb_research/anchor_v2/experiments/lgbm_top13_market_diagnostics.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--summary-output",
        default="mlb_research/anchor_v2/experiments/lgbm_top13_market_diagnostics.md",
        help="Markdown summary output path.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260502)
    args = parser.parse_args()

    results = run_diagnostics(args.model_config, args.bootstrap_samples, args.seed)
    atomic_write_json(args.output, results)
    Path(args.summary_output).write_text(render_markdown(results))


if __name__ == "__main__":
    main()
