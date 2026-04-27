"""Unit tests for the isotonic/sigmoid calibration wrapper added for Run 3 prep.

Tests exercise TimeAwareCalibrated directly with synthetic data so they stay
fast and do not depend on the frozen MLB anchor CSV. The parity test with
TimeAwareCalibratedGBM is the backward-compat guarantee: the new wrapper must
reproduce the frozen class bit-exactly when configured as sigmoid+sklearn-GBM.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlb_research" / "anchor"))

from anchor_eval import (  # noqa: E402
    TimeAwareCalibrated,
    TimeAwareCalibratedGBM,
    build_estimator_factory,
)


def _synthetic(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    logits = X["f0"].to_numpy() + 0.5 * X["f1"].to_numpy()
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < p).astype(int)
    return X, y


def test_sigmoid_wrapper_matches_frozen_gbm_class():
    X, y = _synthetic(n=400, seed=42)

    frozen = TimeAwareCalibratedGBM(
        n_estimators=30, max_depth=2, learning_rate=0.05, random_state=7
    )
    frozen.fit(X, y)
    p_frozen = frozen.predict_proba(X)[:, 1]

    wrap = TimeAwareCalibrated(
        base_factory=lambda: GradientBoostingClassifier(
            n_estimators=30, max_depth=2, learning_rate=0.05, random_state=7
        ),
        method="sigmoid",
    )
    wrap.fit(X, y)
    p_wrap = wrap.predict_proba(X)[:, 1]

    np.testing.assert_allclose(p_wrap, p_frozen, atol=1e-10)


def test_isotonic_wrapper_runs_and_returns_valid_probs():
    X, y = _synthetic(n=500, seed=1)
    wrap = TimeAwareCalibrated(
        base_factory=lambda: GradientBoostingClassifier(
            n_estimators=30, max_depth=2, random_state=0
        ),
        method="isotonic",
    )
    wrap.fit(X, y)
    p = wrap.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert wrap.calibrator_ is not None
    assert wrap.calibration_rows_ > 0


def test_wrapper_skips_calibration_below_min_rows():
    X, y = _synthetic(n=100, seed=2)
    wrap = TimeAwareCalibrated(
        base_factory=lambda: GradientBoostingClassifier(
            n_estimators=20, max_depth=2, random_state=0
        ),
        method="isotonic",
        min_calibration_rows=200,
    )
    wrap.fit(X, y)
    assert wrap.calibrator_ is None
    assert wrap.calibration_rows_ == 0
    assert wrap.calibrator_source_ == "skipped_thin_holdout"
    p = wrap.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)


def test_wrapper_calibrates_when_total_clears_min_calibration_rows():
    # The new wrapper's gate now mirrors the frozen TimeAwareCalibratedGBM:
    # calibrate iff len >= min_calibration_rows + class-balance OK. The
    # earlier min_holdout_rows floor was removed for cross-path comparison
    # fairness against the frozen baseline.
    X, y = _synthetic(n=250, seed=11)
    wrap = TimeAwareCalibrated(
        base_factory=lambda: GradientBoostingClassifier(
            n_estimators=20, max_depth=2, random_state=0
        ),
        method="sigmoid",
        min_calibration_rows=200,
    )
    wrap.fit(X, y)
    assert wrap.calibrator_ is not None
    assert wrap.calibrator_source_ == "holdout"


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="sigmoid|isotonic"):
        TimeAwareCalibrated(base_factory=lambda: GradientBoostingClassifier(), method="bogus")


def test_build_factory_routes_default_sklearn_sigmoid_to_frozen_class():
    config = {"calibrated": True}
    factory = build_estimator_factory(config)
    est = factory()
    assert isinstance(est, TimeAwareCalibratedGBM)


def test_build_factory_routes_sklearn_isotonic_to_wrapper():
    config = {"calibrated": True, "calibration_method": "isotonic"}
    factory = build_estimator_factory(config)
    est = factory()
    assert isinstance(est, TimeAwareCalibrated)
    assert est.method == "isotonic"


def test_build_factory_routes_lgbm_calibrated_to_wrapper():
    pytest.importorskip("lightgbm")
    config = {
        "model_family": "lightgbm",
        "calibrated": True,
        "hyperparams": {"n_estimators": 50, "max_depth": 1, "learning_rate": 0.05, "random_state": 42},
    }
    factory = build_estimator_factory(config)
    est = factory()
    assert isinstance(est, TimeAwareCalibrated)
    assert est.method == "sigmoid"


def test_build_factory_rejects_unknown_method():
    config = {"calibration_method": "nonsense"}
    with pytest.raises(SystemExit):
        build_estimator_factory(config)


def test_build_factory_rejects_inert_method_when_calibrated_false():
    # If calibrated=false, the method key has no effect. Recording it would
    # mis-label the experiment in the ledger. Adversarial review caught this.
    config = {"calibrated": False, "calibration_method": "isotonic"}
    with pytest.raises(SystemExit, match="calibrated=false"):
        build_estimator_factory(config)


def test_build_factory_rejects_inert_method_when_target_margin():
    # target=margin always implies calibrated=false (calibrated=true is rejected
    # earlier), so the calibrated=false branch fires first when both are set --
    # but the rejection still happens, which is the guarantee the test exists
    # to enforce. The inert check rejects regardless of which branch trips.
    config = {"target": "margin", "calibrated": False, "calibration_method": "sigmoid"}
    with pytest.raises(SystemExit, match="calibration_method"):
        build_estimator_factory(config)


def test_build_factory_allows_omitted_calibration_method_with_calibrated_false():
    # Default calibration_method ("sigmoid") is implicit, not an active claim,
    # so it must NOT trigger the inert-method rejection when omitted.
    config = {"calibrated": False}
    factory = build_estimator_factory(config)
    factory()  # should not raise


def test_lgbm_isotonic_end_to_end():
    pytest.importorskip("lightgbm")
    config = {
        "model_family": "lightgbm",
        "calibrated": True,
        "calibration_method": "isotonic",
        "hyperparams": {"n_estimators": 50, "max_depth": 1, "learning_rate": 0.05, "random_state": 42},
    }
    factory = build_estimator_factory(config)
    est = factory()
    X, y = _synthetic(n=400, seed=5)
    est.fit(X, y)
    p = est.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert est.calibrator_ is not None
