"""Unit tests for the keep/revert gate in run_experiment.

Exercise `recommendation()` directly with synthetic optimizer dicts so the
tests don't touch the anchor evaluation or the real ledger.

The earlier cumulative-delta secondary gate was removed after adversarial
review pointed out that its 0.015 threshold was at the search-noise floor
on the same window. The runner now uses only the primary 0.010 marginal
floor; multi-change candidates that the primary floor rejects can be
promoted by a deliberate human-override row in the ledger.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlb_research"))

from run_experiment import (  # noqa: E402
    MAX_FALLBACK_SHARE_FOR_KEEP,
    MIN_BRIER_DELTA_FOR_KEEP,
    MIN_N_HC_FOR_KEEP,
    recommendation,
    running_best_optimizer,
)


BASELINE_0_2553 = {"brier": 0.2553, "roi_units": 54.55}
RUN2_BEST_0_2420 = {"brier": 0.2420, "roi_units": 124.64}


def _opt(brier, roi=100.0, n_hc=1100):
    return {"brier": brier, "roi_units": roi, "n_high_conf": n_hc}


def test_no_prior_best_keeps_as_baseline():
    rec = recommendation(_opt(0.30), best_opt=None, roi_regression_cap=3.0)
    assert rec.startswith("KEEP")
    assert "baseline" in rec.lower()


def test_primary_gate_passes_when_marginal_above_floor():
    # Run 2 exp 15: 0.2553 -> 0.2420, Δ=-0.0133 vs running best (=baseline).
    rec = recommendation(
        _opt(0.2420, roi=124.64, n_hc=1143),
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("KEEP")


def test_sub_floor_marginal_reverts():
    # Run 2 exp 21 case: marginal 0.0018 < 0.010. Must REVERT now that the
    # cumulative gate is gone.
    rec = recommendation(
        _opt(0.2402, roi=156.36, n_hc=1159),
        best_opt=RUN2_BEST_0_2420,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "within-noise" in rec or "improved by only" in rec


def test_no_improvement_reverts():
    rec = recommendation(
        _opt(RUN2_BEST_0_2420["brier"]),  # exact tie
        best_opt=RUN2_BEST_0_2420,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "did not improve" in rec


def test_brier_regression_reverts():
    rec = recommendation(
        _opt(0.260),  # worse than running best 0.2420
        best_opt=RUN2_BEST_0_2420,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")


def test_roi_regression_cap_blocks_even_with_strong_brier():
    rec = recommendation(
        _opt(0.2420, roi=40.0, n_hc=1100),  # Brier crosses floor, ROI down 84U
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "opt_roi regressed" in rec


def test_low_n_hc_reverts_even_with_big_brier_improvement():
    rec = recommendation(
        _opt(0.2400, roi=200.0, n_hc=300),  # n_hc < MIN_N_HC_FOR_KEEP=500
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "n_high_conf=300" in rec
    assert str(MIN_N_HC_FOR_KEEP) in rec


def test_missing_metric_reverts():
    rec = recommendation(
        {"brier": None, "roi_units": None, "n_high_conf": 0},
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "metric missing" in rec


def test_running_best_picks_lowest_brier_among_baseline_and_kept():
    rows = [
        {"status": "baseline", "opt_brier": "0.2553", "opt_roi": "54.55", "commit": "a"},
        {"status": "kept", "opt_brier": "0.2420", "opt_roi": "124.64", "commit": "b"},
        {"status": "reverted", "opt_brier": "0.2200", "opt_roi": "150.00", "commit": "c"},
    ]
    b = running_best_optimizer(rows)
    assert b["brier"] == pytest.approx(0.2420)


def test_constants_have_sensible_relationship():
    assert MIN_BRIER_DELTA_FOR_KEEP > 0
    assert MIN_N_HC_FOR_KEEP > 0
    assert 0 < MAX_FALLBACK_SHARE_FOR_KEEP <= 1


def test_fallback_share_blocks_keep_when_calibration_mostly_skipped():
    # Strong primary-gate-clearing Brier improvement (Δ=0.013), good ROI
    # and n_hc, but the calibration mechanism was skipped on most folds.
    # The label "isotonic" or "sigmoid" would mis-represent what was run.
    diag = {
        "n_folds_trained": 10,
        "calibrator_source_counts": {"holdout": 4, "skipped_thin_holdout": 6},
        "sigma_source_counts": {},
    }
    rec = recommendation(
        _opt(0.2420, roi=120.0, n_hc=1100),
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
        diagnostics=diag,
    )
    assert rec.startswith("REVERT")
    assert "calibration skipped" in rec
    assert "fallback share" in rec


def test_fallback_share_blocks_keep_when_margin_sigma_falls_back():
    diag = {
        "n_folds_trained": 10,
        "calibrator_source_counts": {},
        "sigma_source_counts": {"holdout": 4, "std_of_y_fallback": 6},
    }
    rec = recommendation(
        _opt(0.2420, roi=120.0, n_hc=1100),
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
        diagnostics=diag,
    )
    assert rec.startswith("REVERT")
    assert "margin sigma" in rec


def test_fallback_share_below_threshold_does_not_block_keep():
    # 1/10 = 10% fallback < 20% threshold; primary gate decision stands.
    diag = {
        "n_folds_trained": 10,
        "calibrator_source_counts": {"holdout": 9, "skipped_thin_holdout": 1},
        "sigma_source_counts": {},
    }
    rec = recommendation(
        _opt(0.2420, roi=120.0, n_hc=1100),
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
        diagnostics=diag,
    )
    assert rec.startswith("KEEP")


def test_fallback_share_with_no_diagnostics_is_no_op():
    # When neither calibration nor margin path is active (e.g. the frozen
    # baseline), source counts are empty and the gate is a no-op.
    diag = {
        "n_folds_trained": 10,
        "calibrator_source_counts": {},
        "sigma_source_counts": {},
    }
    rec = recommendation(
        _opt(0.2420, roi=120.0, n_hc=1100),
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
        diagnostics=diag,
    )
    assert rec.startswith("KEEP")


def test_fallback_share_with_none_diagnostics_is_no_op():
    rec = recommendation(
        _opt(0.2420, roi=120.0, n_hc=1100),
        best_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
        diagnostics=None,
    )
    assert rec.startswith("KEEP")


def test_cmd_run_rejects_multiple_baseline_rows(tmp_path):
    # The runner's stop conditions anchor on the most recent baseline/kept
    # row, so multiple baseline rows would silently shift which one governs.
    # cmd_run must reject that state up front. Test runs against an isolated
    # temp ledger via the MLB_RESEARCH_RESULTS_TSV env var so we never touch
    # the real checked-in results.tsv -- adversarial review caught the
    # earlier version of this test mutating the repo in place.
    import os
    import subprocess

    runner = REPO_ROOT / "mlb_research" / "run_experiment.py"
    real_tsv = REPO_ROOT / "mlb_research" / "results.tsv"

    header = real_tsv.read_text().splitlines()[0]
    fake_row1 = "20260101T000000Z\taaaaaaa\t0.255300\t0.720000\t50.00\t0.5500\t800\t1400\t0.240000\t80.00\t480\t0.190000\t40.00\t78\tbaseline\tbaseline\tbaseline 1\tx\ty"
    fake_row2 = "20260102T000000Z\tbbbbbbb\t0.250000\t0.700000\t60.00\t0.5600\t820\t1400\t0.240000\t80.00\t480\t0.190000\t40.00\t78\tbaseline\tbaseline\tbaseline 2\tx\ty"

    fake_tsv = tmp_path / "results.tsv"
    fake_tsv.write_text("\n".join([header, fake_row1, fake_row2]) + "\n")
    fake_experiments = tmp_path / "experiments"

    env = {
        **os.environ,
        "MLB_RESEARCH_RESULTS_TSV": str(fake_tsv),
        "MLB_RESEARCH_EXPERIMENTS_DIR": str(fake_experiments),
    }
    result = subprocess.run(
        [sys.executable, str(runner), "run",
         "--config", str(REPO_ROOT / "mlb_research" / "configs" / "baseline.json"),
         "--change-type", "features",
         "--description", "should be rejected -- two baselines"],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Multiple baseline rows" in combined
    # Sanity: the real ledger was not touched.
    assert real_tsv.read_text().splitlines()[0] == header
    assert "baseline 1" not in real_tsv.read_text()
