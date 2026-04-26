"""Tests for unknown-config-key rejection in anchor_eval.

Protects autonomous runs from typos silently falling back to defaults: an
agent that writes `"model_familyy": "lightgbm"` should fail fast, not
accidentally re-run the sklearn GBM baseline labeled as a LightGBM trial.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlb_research" / "anchor"))

from anchor_eval import (  # noqa: E402
    VALID_HYPERPARAM_KEYS,
    VALID_TOP_LEVEL_CONFIG_KEYS,
    validate_config_keys,
)

ANCHOR_EVAL = REPO_ROOT / "mlb_research" / "anchor" / "anchor_eval.py"


def test_known_keys_accepted():
    config = {
        "features": ["is_home"],
        "hyperparams": {"n_estimators": 100, "subsample": 0.8},
        "model_family": "lightgbm",
        "calibrated": False,
        "calibration_method": "sigmoid",
        "target": "margin",
    }
    validate_config_keys(config)  # should not raise


def test_underscore_keys_treated_as_comments():
    config = {"_description": "anything", "_notes": "see linear issue", "target": "home_win"}
    validate_config_keys(config)


def test_unknown_top_level_key_rejected():
    with pytest.raises(SystemExit, match="Unknown top-level config key"):
        validate_config_keys({"model_familyy": "lightgbm"})


def test_unknown_hyperparam_key_rejected():
    with pytest.raises(SystemExit, match="Unknown hyperparams key"):
        validate_config_keys({"hyperparams": {"n_estimator": 150}})


def test_common_typo_caught():
    # Common autonomous agent typo: plural/singular confusion.
    with pytest.raises(SystemExit, match="n_estimator"):
        validate_config_keys({"hyperparams": {"n_estimator": 150, "max_depth": 2}})


def test_all_baseline_hyperparam_keys_are_valid():
    # The baseline.json keys must all be in the whitelist. If someone adds
    # a new knob to the config but forgets the whitelist, the canonical
    # baseline eval would start failing -- this test catches that.
    baseline = json.loads((REPO_ROOT / "mlb_research" / "configs" / "baseline.json").read_text())
    validate_config_keys(baseline)


def test_all_experiment_configs_are_valid():
    # Sweep every config JSON actually used in past experiments. None should
    # trip the new validator (that would mean a past ledger row used a
    # misspelled key and succeeded via default-fallback; worth knowing).
    cfg_dir = REPO_ROOT / "mlb_research" / "configs"
    bad = []
    for p in cfg_dir.glob("*.json"):
        try:
            validate_config_keys(json.loads(p.read_text()))
        except SystemExit as e:
            bad.append(f"{p.name}: {e}")
    assert not bad, "Historical configs failed validation:\n" + "\n".join(bad)


def test_validator_runs_via_cli(tmp_path):
    # End-to-end: a bogus config dies before the walk-forward eval kicks off.
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"model_familyy": "lightgbm"}))
    result = subprocess.run(
        [sys.executable, str(ANCHOR_EVAL), "--model-config", str(p),
         "--output", str(tmp_path / "r.json")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode != 0
    assert "Unknown top-level config key" in result.stderr + result.stdout


def test_valid_key_whitelists_are_not_empty():
    # Sanity: guarding against accidental wipe of either whitelist.
    assert "target" in VALID_TOP_LEVEL_CONFIG_KEYS
    assert "n_estimators" in VALID_HYPERPARAM_KEYS
    assert len(VALID_TOP_LEVEL_CONFIG_KEYS) >= 6
    assert len(VALID_HYPERPARAM_KEYS) >= 8


def test_active_keys_for_baseline_path():
    # Baseline config: sklearn_gbm + calibrated + home_win. Active keys are
    # core + holdout (no sampling for sklearn_gbm).
    from anchor_eval import active_hyperparam_keys
    config = {"model_family": "sklearn_gbm", "calibrated": True, "target": "home_win"}
    keys = active_hyperparam_keys(config)
    assert "n_estimators" in keys and "calibration_fraction" in keys
    assert "subsample" not in keys and "colsample_bytree" not in keys


def test_active_keys_for_lgbm_calibrated():
    from anchor_eval import active_hyperparam_keys
    config = {"model_family": "lightgbm", "calibrated": True, "target": "home_win"}
    keys = active_hyperparam_keys(config)
    assert {"subsample", "colsample_bytree", "calibration_fraction", "min_calibration_rows"} <= keys


def test_active_keys_for_uncalibrated_home_win():
    from anchor_eval import active_hyperparam_keys
    config = {"model_family": "sklearn_gbm", "calibrated": False, "target": "home_win"}
    keys = active_hyperparam_keys(config)
    assert "calibration_fraction" not in keys
    assert "min_calibration_rows" not in keys


def test_active_keys_for_margin_target():
    # Margin uses holdout keys for residual sigma estimation regardless
    # of calibrated (which must be false anyway for target=margin).
    from anchor_eval import active_hyperparam_keys
    config = {"model_family": "sklearn_gbm", "calibrated": False, "target": "margin"}
    keys = active_hyperparam_keys(config)
    assert {"calibration_fraction", "min_calibration_rows"} <= keys


def test_validator_rejects_subsample_on_sklearn_gbm():
    # Adversarial review: sklearn_gbm doesn't accept subsample/colsample.
    # Recording them would mis-label the experiment.
    config = {
        "model_family": "sklearn_gbm",
        "calibrated": True,
        "target": "home_win",
        "hyperparams": {"n_estimators": 100, "subsample": 0.5},
    }
    with pytest.raises(SystemExit, match="Inert hyperparams"):
        validate_config_keys(config)


def test_validator_rejects_calibration_fraction_when_uncalibrated_home_win():
    # calibration_fraction has no effect when calibrated=false on home_win.
    config = {
        "model_family": "sklearn_gbm",
        "calibrated": False,
        "target": "home_win",
        "hyperparams": {"n_estimators": 100, "calibration_fraction": 0.2},
    }
    with pytest.raises(SystemExit, match="Inert hyperparams"):
        validate_config_keys(config)


def test_validator_accepts_subsample_on_lgbm():
    config = {
        "model_family": "lightgbm",
        "calibrated": False,
        "target": "home_win",
        "hyperparams": {"n_estimators": 100, "subsample": 0.5, "colsample_bytree": 0.8},
    }
    validate_config_keys(config)  # should not raise


def test_validator_rejects_string_calibrated_false():
    # The infamous string-bool case: JSON {"calibrated": "false"} is truthy
    # in Python and would silently route the calibrated path while the
    # archived config and _meta still claimed calibrated=false. Adversarial
    # review caught this exact ledger-truthfulness failure mode.
    with pytest.raises(SystemExit, match="must be a JSON boolean"):
        validate_config_keys({"calibrated": "false"})


def test_validator_rejects_string_calibrated_true():
    with pytest.raises(SystemExit, match="must be a JSON boolean"):
        validate_config_keys({"calibrated": "true"})


def test_validator_rejects_int_calibrated():
    with pytest.raises(SystemExit, match="must be a JSON boolean"):
        validate_config_keys({"calibrated": 1})


def test_validator_rejects_string_n_estimators():
    config = {"hyperparams": {"n_estimators": "100"}}
    with pytest.raises(SystemExit, match="must be an integer"):
        validate_config_keys(config)


def test_validator_rejects_float_for_int_field():
    # n_estimators must be int (a float silently truncates in some libs,
    # making the archived config non-reproducible).
    config = {"hyperparams": {"n_estimators": 100.5}}
    with pytest.raises(SystemExit, match="must be an integer"):
        validate_config_keys(config)


def test_validator_rejects_string_learning_rate():
    config = {"hyperparams": {"learning_rate": "0.05"}}
    with pytest.raises(SystemExit, match="must be a number"):
        validate_config_keys(config)


def test_validator_rejects_bool_for_numeric_hyperparam():
    # bool is a subclass of int in Python -- explicit rejection prevents
    # `n_estimators: true` silently meaning n_estimators=1.
    config = {"hyperparams": {"n_estimators": True}}
    with pytest.raises(SystemExit, match="bool"):
        validate_config_keys(config)


def test_validator_rejects_non_dict_hyperparams():
    with pytest.raises(SystemExit, match="must be a JSON object"):
        validate_config_keys({"hyperparams": ["n_estimators=100"]})


def test_validator_rejects_non_list_features():
    with pytest.raises(SystemExit, match="must be a JSON array"):
        validate_config_keys({"features": "is_home,rest_days"})


def test_validator_accepts_int_for_float_field():
    # learning_rate=0 is sometimes meaningful; an int there should be fine.
    config = {
        "model_family": "sklearn_gbm",
        "calibrated": False,
        "target": "home_win",
        "hyperparams": {"n_estimators": 100, "learning_rate": 1},
    }
    validate_config_keys(config)  # should not raise
