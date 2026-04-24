"""Unit tests for the margin-regression + Normal CDF target path.

Exercises MarginCDFRegressor directly with synthetic data so the tests stay
fast and don't touch the frozen MLB anchor CSV. Integration with the full
anchor eval is covered by the baseline reproducibility check (run manually)
and a lightweight end-to-end smoke test through build_estimator_factory.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlb_research" / "anchor"))

from anchor_eval import (  # noqa: E402
    MarginCDFRegressor,
    build_estimator_factory,
)


def _synthetic_margin(n=500, seed=0, true_sigma=3.0):
    """Generate (features, margin) with a linear signal + Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    mu = 1.5 * X["f0"].to_numpy() + 0.7 * X["f1"].to_numpy()
    margin = mu + rng.normal(scale=true_sigma, size=n)
    return X, margin


def test_margin_regressor_returns_valid_probs():
    X, y = _synthetic_margin(n=500, seed=1)
    reg = MarginCDFRegressor(
        base_factory=lambda: GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=0),
    )
    reg.fit(X, y)
    p = reg.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_margin_regressor_sigma_is_reasonable():
    # With true_sigma=3.0 and a decent regressor, residual std should be near 3.
    X, y = _synthetic_margin(n=600, seed=2, true_sigma=3.0)
    reg = MarginCDFRegressor(
        base_factory=lambda: GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0),
    )
    reg.fit(X, y)
    assert 2.0 < reg.sigma_ < 5.0, f"sigma_={reg.sigma_}"
    assert reg.residual_rows_ > 0


def test_margin_regressor_prob_monotone_in_mu():
    # For the same sigma, larger predicted margin must map to larger P(home).
    X, y = _synthetic_margin(n=400, seed=3)
    reg = MarginCDFRegressor(
        base_factory=lambda: GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=0),
    )
    reg.fit(X, y)
    # Query two synthetic rows with very different predicted margins by
    # manipulating the dominant feature.
    query = pd.DataFrame(
        {"f0": [-2.0, 2.0], "f1": [0.0, 0.0], "f2": [0.0, 0.0], "f3": [0.0, 0.0]}
    )
    p = reg.predict_proba(query)[:, 1]
    assert p[0] < 0.5 < p[1], p


def test_margin_regressor_applies_min_sigma_floor():
    # Fit noise-free data; residual std would be 0 but we floor at min_sigma.
    n = 400
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=[f"f{i}" for i in range(3)])
    y = 2.0 * X["f0"].to_numpy()  # perfectly explained by f0
    reg = MarginCDFRegressor(
        base_factory=lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=0),
        min_sigma=0.5,
    )
    reg.fit(X, y)
    assert reg.sigma_ >= 0.5


def test_margin_regressor_falls_back_to_in_sample_residuals_below_min():
    X, y = _synthetic_margin(n=80, seed=4)
    reg = MarginCDFRegressor(
        base_factory=lambda: GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=0),
        min_residual_rows=200,
    )
    reg.fit(X, y)
    assert reg.residual_rows_ == 0
    assert np.isfinite(reg.sigma_)


def test_build_factory_routes_margin_target_to_regressor():
    config = {"target": "margin", "calibrated": False}
    factory = build_estimator_factory(config)
    est = factory()
    assert isinstance(est, MarginCDFRegressor)


def test_build_factory_rejects_calibrated_margin():
    config = {"target": "margin", "calibrated": True}
    with pytest.raises(SystemExit, match="target='margin'"):
        build_estimator_factory(config)


def test_build_factory_rejects_unknown_target():
    config = {"target": "run_line_cover"}
    with pytest.raises(SystemExit, match="Unsupported target"):
        build_estimator_factory(config)


def test_build_factory_home_win_target_still_routes_to_classifier():
    from anchor_eval import TimeAwareCalibratedGBM
    config = {"target": "home_win", "calibrated": True}
    factory = build_estimator_factory(config)
    est = factory()
    assert isinstance(est, TimeAwareCalibratedGBM)


def test_lgbm_margin_end_to_end():
    pytest.importorskip("lightgbm")
    config = {
        "model_family": "lightgbm",
        "target": "margin",
        "calibrated": False,
        "hyperparams": {
            "n_estimators": 50, "max_depth": 2, "learning_rate": 0.05, "random_state": 42
        },
    }
    factory = build_estimator_factory(config)
    est = factory()
    assert isinstance(est, MarginCDFRegressor)
    X, y = _synthetic_margin(n=400, seed=5)
    est.fit(X, y)
    p = est.predict_proba(X)[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_xgboost_margin_end_to_end():
    pytest.importorskip("xgboost")
    config = {
        "model_family": "xgboost",
        "target": "margin",
        "calibrated": False,
        "hyperparams": {
            "n_estimators": 50, "max_depth": 2, "learning_rate": 0.05, "random_state": 42
        },
    }
    factory = build_estimator_factory(config)
    est = factory()
    assert isinstance(est, MarginCDFRegressor)
    X, y = _synthetic_margin(n=400, seed=6)
    est.fit(X, y)
    p = est.predict_proba(X)[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0
