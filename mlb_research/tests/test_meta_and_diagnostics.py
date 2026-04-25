"""Tests for ledger-truthfulness fixes from the second adversarial review.

Two guarantees:
- `_meta.calibration_method` is None whenever the calibration path was not
  actually exercised (caught silent experiment-labeling bug).
- `diagnostics.calibrator_source_counts` and `diagnostics.sigma_source_counts`
  reflect per-fold wrapper behavior so silent fallbacks (thin holdouts,
  std-of-y sigma) are observable in the metrics JSON.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_EVAL = REPO_ROOT / "mlb_research" / "anchor" / "anchor_eval.py"


def _run_eval(config_path: Path, out_path: Path):
    subprocess.run(
        [sys.executable, str(ANCHOR_EVAL),
         "--model-config", str(config_path), "--output", str(out_path)],
        check=True, cwd=REPO_ROOT, capture_output=True,
    )
    return json.loads(out_path.read_text())


def test_meta_calibration_method_is_null_when_calibrated_false(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "model_family": "sklearn_gbm",
        "calibrated": False,
        "target": "home_win",
        "hyperparams": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05,
                         "random_state": 42, "calibration_fraction": 0.2,
                         "min_calibration_rows": 200},
    }))
    r = _run_eval(cfg, tmp_path / "r.json")
    assert r["_meta"]["calibration_method"] is None
    assert r["_meta"]["calibrated"] is False


def test_meta_calibration_method_is_null_when_target_margin(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "model_family": "sklearn_gbm",
        "calibrated": False,
        "target": "margin",
        "hyperparams": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05,
                         "random_state": 42, "calibration_fraction": 0.2,
                         "min_calibration_rows": 200},
    }))
    r = _run_eval(cfg, tmp_path / "r.json")
    assert r["_meta"]["calibration_method"] is None
    assert r["_meta"]["target"] == "margin"


def test_meta_calibration_method_is_recorded_when_actually_active(tmp_path):
    # Default baseline path: calibrated sklearn GBM with sigmoid.
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "model_family": "sklearn_gbm",
        "calibrated": True,
        "target": "home_win",
        "hyperparams": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05,
                         "random_state": 42, "calibration_fraction": 0.2,
                         "min_calibration_rows": 200},
    }))
    r = _run_eval(cfg, tmp_path / "r.json")
    assert r["_meta"]["calibration_method"] == "sigmoid"


def test_diagnostics_record_calibrator_source_counts(tmp_path):
    # LGBM + isotonic calibration runs through TimeAwareCalibrated, which sets
    # calibrator_source_ on every fold. Counts must surface in diagnostics.
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "model_family": "lightgbm",
        "calibrated": True,
        "calibration_method": "isotonic",
        "target": "home_win",
        "hyperparams": {"n_estimators": 30, "max_depth": 1, "learning_rate": 0.05,
                         "random_state": 42, "subsample": 0.8, "colsample_bytree": 0.8,
                         "calibration_fraction": 0.2, "min_calibration_rows": 200},
    }))
    r = _run_eval(cfg, tmp_path / "r.json")
    diag = r["_meta"]["diagnostics"]["optimizer"]
    counts = diag["calibrator_source_counts"]
    # At least one of the two source labels must appear.
    assert counts, "Expected non-empty calibrator_source_counts for LGBM+isotonic"
    assert all(k in {"holdout", "skipped_thin_holdout"} for k in counts)
    assert sum(counts.values()) == diag["n_folds_trained"]


def test_diagnostics_record_sigma_source_counts(tmp_path):
    # Margin target runs through MarginCDFRegressor which sets sigma_source_.
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "model_family": "sklearn_gbm",
        "calibrated": False,
        "target": "margin",
        "hyperparams": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05,
                         "random_state": 42, "calibration_fraction": 0.2,
                         "min_calibration_rows": 200},
    }))
    r = _run_eval(cfg, tmp_path / "r.json")
    diag = r["_meta"]["diagnostics"]["optimizer"]
    counts = diag["sigma_source_counts"]
    assert counts, "Expected non-empty sigma_source_counts for margin target"
    assert all(k in {"holdout", "std_of_y_fallback"} for k in counts)
    assert sum(counts.values()) == diag["n_folds_trained"]


def test_diagnostics_source_counts_empty_for_baseline_path(tmp_path):
    # Frozen TimeAwareCalibratedGBM does NOT expose source attributes, so the
    # counts must be empty. Confirms the getattr-with-default plumbing.
    cfg = REPO_ROOT / "mlb_research" / "configs" / "baseline.json"
    r = _run_eval(cfg, tmp_path / "r.json")
    diag = r["_meta"]["diagnostics"]["optimizer"]
    assert diag["calibrator_source_counts"] == {}
    assert diag["sigma_source_counts"] == {}
