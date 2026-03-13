"""Tests for model_win.py helper behavior."""

import joblib
import pandas as pd
import pytest

from model_win import (
    FEATURES_NO_LINE,
    FEATURES_WITH_LINE,
    MENS_FEATURES_NO_LINE,
    MENS_FEATURES_WITH_LINE,
    _prepare_home_rows,
    get_win_feature_list,
    load_win_model_bundle,
    predict_home_win_prob,
)


class DummyModel:
    """Simple deterministic model stub with sklearn-like API."""

    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict_proba(self, X):
        return [[1.0 - self.prob, self.prob] for _ in range(len(X))]


def test_predict_uses_with_line_when_spread_present():
    bundle = {
        "model_no_line": DummyModel(0.55),
        "features_no_line": FEATURES_NO_LINE,
        "model_with_line": DummyModel(0.72),
        "features_with_line": FEATURES_WITH_LINE,
    }
    prob, variant = predict_home_win_prob({"spread": -3.5}, bundle)
    assert variant == "with_line"
    assert prob == pytest.approx(0.72)


def test_predict_falls_back_to_no_line_when_spread_absent():
    bundle = {
        "model_no_line": DummyModel(0.61),
        "features_no_line": FEATURES_NO_LINE,
        "model_with_line": DummyModel(0.72),
        "features_with_line": FEATURES_WITH_LINE,
    }
    prob, variant = predict_home_win_prob({"spread": 0.0}, bundle)
    assert variant == "no_line"
    assert prob == pytest.approx(0.61)


def test_predict_falls_back_to_no_line_when_with_line_missing():
    bundle = {
        "model_no_line": DummyModel(0.58),
        "features_no_line": FEATURES_NO_LINE,
        "model_with_line": None,
        "features_with_line": FEATURES_WITH_LINE,
    }
    prob, variant = predict_home_win_prob({"spread": -4.0}, bundle)
    assert variant == "no_line"
    assert prob == pytest.approx(0.58)


def test_predict_respects_allow_with_line_false():
    bundle = {
        "model_no_line": DummyModel(0.59),
        "features_no_line": FEATURES_NO_LINE,
        "model_with_line": DummyModel(0.77),
        "features_with_line": FEATURES_WITH_LINE,
    }
    prob, variant = predict_home_win_prob({"spread": -4.0}, bundle, allow_with_line=False)
    assert variant == "no_line"
    assert prob == pytest.approx(0.59)


def test_predict_raises_without_no_line_model():
    bundle = {
        "model_no_line": None,
        "features_no_line": FEATURES_NO_LINE,
        "model_with_line": None,
        "features_with_line": FEATURES_WITH_LINE,
    }
    with pytest.raises(ValueError, match="No available no-line model"):
        predict_home_win_prob({"spread": -1.5}, bundle)


def test_load_bundle_supports_raw_legacy_model(tmp_path):
    legacy_model = DummyModel(0.64)
    p = tmp_path / "legacy.pkl"
    joblib.dump(legacy_model, p)

    bundle = load_win_model_bundle(path=str(p))

    assert bundle["model_no_line"] is not None
    assert bundle["model_with_line"] is None
    assert bundle["features_no_line"] == MENS_FEATURES_NO_LINE


def test_prepare_home_rows_handles_missing_off_rating_columns():
    df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "is_home": [1],
            "team_score": [80],
            "opp_score": [70],
            "spread": [-4.5],
        }
    )

    result = _prepare_home_rows(df)

    assert result.loc[result.index[0], "off_rating_gap"] == pytest.approx(0.0)


def test_mens_win_feature_lists_drop_dead_advanced_inputs():
    assert get_win_feature_list("mens", with_line=False) == MENS_FEATURES_NO_LINE
    assert get_win_feature_list("mens", with_line=True) == MENS_FEATURES_WITH_LINE
    assert "diff_eFG" not in MENS_FEATURES_NO_LINE
