import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlb_research" / "anchor"))
from anchor_eval import summarize


def test_summarize_moneyline_roi_scores_selected_side_prices():
    preds = pd.DataFrame(
        {
            "prob_home": [0.60, 0.40, 0.61],
            "target": [1, 0, 0],
            "conf": [0.60, 0.60, 0.61],
            "home_moneyline": [-150, 120, -110],
            "away_moneyline": [130, -140, -110],
        }
    )

    result = summarize(preds, high_conf_threshold=0.53, roi_mode="moneyline")

    # Win home at -150: +0.6667; win away at -140: +0.7143; lose home: -1.
    assert result["roi_units"] == pytest.approx((100 / 150) + (100 / 140) - 1)
    assert result["n_roi_priced"] == 3
    assert result["n_roi_missing_price"] == 0
    assert result["n_high_conf"] == 3


def test_summarize_moneyline_roi_reports_missing_selected_prices():
    preds = pd.DataFrame(
        {
            "prob_home": [0.60, 0.40],
            "target": [1, 0],
            "conf": [0.60, 0.60],
            "home_moneyline": [-150, 120],
            "away_moneyline": [np.nan, np.nan],
        }
    )

    result = summarize(preds, high_conf_threshold=0.53, roi_mode="moneyline")

    assert result["roi_units"] == pytest.approx(100 / 150)
    assert result["n_roi_priced"] == 1
    assert result["n_roi_missing_price"] == 1
    assert result["n_high_conf"] == 2
