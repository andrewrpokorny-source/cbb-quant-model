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
