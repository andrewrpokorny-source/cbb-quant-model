import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mlb_research.anchor_v2 import market_candidate_diagnostics as diag


def _prediction_frame(probs: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-03"]),
            "prob_home": probs,
            "target": [1, 0, 1],
            "conf": [max(p, 1 - p) for p in probs],
            "home_moneyline": [-120, -110, 140],
            "away_moneyline": [110, 100, -150],
        }
    )


def test_paired_prediction_frame_rejects_unmatched_rows():
    model = _prediction_frame([0.60, 0.40, 0.70])
    market = _prediction_frame([0.55, 0.45, 0.65])
    market.loc[1, "target"] = 1

    with pytest.raises(ValueError, match="target"):
        diag.paired_prediction_frame(model, market)


def test_prediction_summary_reports_model_vs_market_deltas():
    frame = diag.paired_prediction_frame(
        _prediction_frame([0.70, 0.30, 0.80]),
        _prediction_frame([0.55, 0.45, 0.52]),
    )

    summary = diag.prediction_summary(frame)

    assert summary["n_games"] == 3
    assert summary["brier_delta"] < 0
    assert summary["log_loss_delta"] < 0
    assert summary["model_n_hc"] == 3
    assert summary["market_n_hc"] == 2
    assert summary["model_hc_accuracy"] == 1.0


def test_bootstrap_deltas_are_seeded_and_shaped():
    frame = diag.paired_prediction_frame(
        _prediction_frame([0.70, 0.30, 0.80]),
        _prediction_frame([0.55, 0.45, 0.52]),
    )

    result = diag.bootstrap_deltas(frame, samples=25, seed=7)

    assert result["samples"] == 25
    assert result["seed"] == 7
    for key in ("brier_delta", "log_loss_delta", "roi_units_delta"):
        assert set(result[key]) == {"mean", "p025", "p975"}
