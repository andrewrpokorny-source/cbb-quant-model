"""Tests for model_win.py helper behavior."""

import joblib
import pytest

from model_win import (
    FEATURES_NO_LINE,
    FEATURES_WITH_LINE,
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


def test_load_bundle_supports_raw_legacy_model(tmp_path):
    legacy_model = DummyModel(0.64)
    p = tmp_path / "legacy.pkl"
    joblib.dump(legacy_model, p)

    bundle = load_win_model_bundle(path=str(p))

    assert bundle["model_no_line"] is not None
    assert bundle["model_with_line"] is None
    assert bundle["features_no_line"] == FEATURES_NO_LINE
