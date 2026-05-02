import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from mlb_research.anchor import anchor_eval
from mlb_research.anchor_v2 import evaluate_direct_market as direct_market


def _market_frame() -> pd.DataFrame:
    rows = []
    for offset in range(60):
        rows.append(
            {
                "date": pd.Timestamp("2025-03-01") + pd.Timedelta(days=offset),
                "is_home": offset % 2,
                "home_win": offset % 2,
                "market_home_no_vig_prob": 0.55,
                "team_moneyline": -120,
                "opp_moneyline": 110,
                "sparse_feature": 1.0,
            }
        )
    for offset, prob in enumerate([0.60, 0.40, 0.62]):
        rows.append(
            {
                "date": pd.Timestamp("2025-05-01") + pd.Timedelta(days=offset),
                "is_home": 1,
                "home_win": int(prob > 0.5),
                "market_home_no_vig_prob": prob,
                "team_moneyline": -125,
                "opp_moneyline": 115,
                "sparse_feature": np.nan if offset == 1 else 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_direct_market_predictions_mirror_weekly_row_mask_and_diagnostics():
    preds, diag = direct_market.direct_market_predictions(
        _market_frame(),
        datetime(2025, 5, 1),
        datetime(2025, 5, 8),
        extra_required_columns=["sparse_feature"],
    )

    assert len(preds) == 2
    assert preds["prob_home"].tolist() == [0.60, 0.62]
    assert preds["conf"].tolist() == [0.60, 0.62]
    assert diag["n_folds_trained"] == 1
    assert diag["pre_dropna_rows"] == 3
    assert diag["post_dropna_rows"] == 2
    assert diag["dropna_loss_rows"] == 1
    assert diag["dropna_loss_share"] == pytest.approx(1 / 3)


def test_direct_market_predictions_reject_out_of_range_probability():
    df = _market_frame()
    df.loc[df["date"] == pd.Timestamp("2025-05-01"), "market_home_no_vig_prob"] = 1.2

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        direct_market.direct_market_predictions(
            df,
            datetime(2025, 5, 1),
            datetime(2025, 5, 8),
        )


def test_direct_market_predictions_reject_non_binary_is_home():
    df = _market_frame()
    df.loc[0, "is_home"] = np.nan

    with pytest.raises(ValueError, match="non-binary or missing is_home"):
        direct_market.direct_market_predictions(
            df,
            datetime(2025, 5, 1),
            datetime(2025, 5, 8),
        )


def test_config_dropna_columns_uses_default_features_when_config_omits_features(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(direct_market, "DEFAULT_MLB_FEATURES", ["foo", "bar"])
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"target": "home_win"}))

    columns, config, features = direct_market.config_dropna_columns(str(cfg))

    assert config["target"] == "home_win"
    assert features == ["foo", "bar"]
    assert columns == ["foo", "bar", "home_win"]


def test_load_manifest_errors_include_manifest_path(tmp_path, monkeypatch):
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps({"sha256": "abc", "row_count": 1}))
    monkeypatch.setattr(anchor_eval, "MANIFEST_PATH", str(bad_manifest))

    with pytest.raises(RuntimeError, match=str(bad_manifest)):
        anchor_eval.load_manifest()
