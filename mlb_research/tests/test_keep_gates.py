"""Unit tests for the primary + secondary keep/revert gates in run_experiment.

Exercise `recommendation()` directly with synthetic optimizer dicts so the
tests don't touch the anchor evaluation or the real ledger. The key scenarios:

- Primary gate: marginal Δ >= 0.010 -> KEEP.
- Secondary (cumulative) gate: marginal sub-floor but cumulative >= 0.015 AND
  marginal >= 0.005 -> KEEP.
- Rejection paths: sub-floor marginal + sub-floor cumulative, ROI regression,
  resolution collapse (n_hc too low), noise-floor marginal even when
  cumulative would otherwise qualify.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlb_research"))

from run_experiment import (  # noqa: E402
    MIN_BRIER_DELTA_FOR_KEEP,
    MIN_CUMULATIVE_DELTA_FOR_KEEP,
    MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP,
    MIN_N_HC_FOR_KEEP,
    baseline_optimizer,
    recommendation,
    running_best_optimizer,
)


# Representative baseline / running-best snapshots from the real ledger.
BASELINE_0_2553 = {"brier": 0.2553, "roi_units": 54.55}
RUN2_BEST_0_2420 = {"brier": 0.2420, "roi_units": 124.64}


def _opt(brier, roi=100.0, n_hc=1100):
    return {"brier": brier, "roi_units": roi, "n_high_conf": n_hc}


def test_no_prior_best_keeps_as_baseline():
    rec = recommendation(_opt(0.30), best_opt=None, baseline_opt=None, roi_regression_cap=3.0)
    assert rec.startswith("KEEP")
    assert "baseline" in rec.lower()


def test_primary_gate_passes_when_marginal_above_floor():
    # Run 2 exp 15: 0.2553 -> 0.2420, Δ=-0.0133 vs running best (=baseline).
    rec = recommendation(
        _opt(0.2420, roi=124.64, n_hc=1143),
        best_opt=BASELINE_0_2553,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("KEEP")
    assert "primary gate" in rec


def test_secondary_gate_passes_stacked_win():
    # Run 2 exp 21: prune-13 + LGBM stumps lands at 0.2402.
    # Marginal vs running best (0.2420) = 0.0018 (sub-floor).
    # Cumulative vs baseline (0.2553) = 0.0151 (>=0.015).
    rec = recommendation(
        _opt(0.2402, roi=156.36, n_hc=1159),
        best_opt=RUN2_BEST_0_2420,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("KEEP"), rec
    assert "cumulative gate" in rec


def test_secondary_gate_rejects_non_positive_marginal_when_gate_available():
    # Tie vs running best, with running best NOT yet past cumulative threshold
    # (so the cumulative gate is still available). Even with cumulative drop
    # qualifying, a zero marginal must REVERT.
    running_best_below_threshold = {"brier": 0.2410, "roi_units": 130.0}  # 0.0143 from baseline
    tied_new = _opt(0.2410, roi=130.0, n_hc=1100)  # marginal 0
    rec = recommendation(
        tied_new,
        best_opt=running_best_below_threshold,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")


def test_secondary_gate_rejects_marginal_below_0001_when_gate_available():
    # Running best 0.0143 from baseline (< 0.015 threshold; gate available).
    # Candidate cumulative = 0.0158 (>= 0.015) but marginal = 0.0005 (< 0.001).
    running_best_below_threshold = {"brier": 0.2410, "roi_units": 130.0}
    new = _opt(0.2405, roi=140.0, n_hc=1100)  # marginal 0.0005, cumulative 0.0148
    rec = recommendation(
        new, best_opt=running_best_below_threshold, baseline_opt=BASELINE_0_2553, roi_regression_cap=3.0
    )
    assert rec.startswith("REVERT")


def test_secondary_gate_accepts_marginal_just_above_0001_when_gate_available():
    # Running best 0.0133 from baseline (gate still available). Candidate
    # cumulative 0.0153, marginal 0.0020. Should KEEP via cumulative gate.
    running_best_below_threshold = {"brier": 0.2420, "roi_units": 124.0}  # Run 2 LGBM stumps
    new = _opt(0.2400, roi=135.0, n_hc=1140)  # marginal 0.0020, cumulative 0.0153
    rec = recommendation(
        new, best_opt=running_best_below_threshold, baseline_opt=BASELINE_0_2553, roi_regression_cap=3.0
    )
    assert rec.startswith("KEEP")
    assert "cumulative gate" in rec


def test_sub_floor_marginal_and_sub_floor_cumulative_reverts():
    # Marginal 0.005 > 0 but < 0.010. Cumulative only 0.008 (< 0.015).
    # Neither gate passes.
    rec = recommendation(
        _opt(0.2545),
        best_opt={"brier": 0.2550, "roi_units": 80.0},
        baseline_opt={"brier": 0.2553, "roi_units": 54.55},
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "primary floor" in rec or "secondary floor" in rec


def test_roi_regression_cap_blocks_even_if_primary_brier_passes():
    # Strong Brier improvement but ROI collapses beyond the 3U cap.
    rec = recommendation(
        _opt(0.2420, roi=40.0, n_hc=1100),  # ROI dropped 84U
        best_opt=BASELINE_0_2553,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "opt_roi regressed" in rec


def test_roi_regression_cap_blocks_secondary_gate():
    # Cumulative gate would pass, but ROI regresses beyond cap.
    rec = recommendation(
        _opt(0.2402, roi=50.0, n_hc=1100),  # baseline ROI was 124.64 at running best
        best_opt=RUN2_BEST_0_2420,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")


def test_low_n_hc_reverts_even_with_big_brier_improvement():
    # Resolution-collapse win: strong Brier, but only 300 high-conf picks.
    rec = recommendation(
        _opt(0.2400, roi=200.0, n_hc=300),
        best_opt=BASELINE_0_2553,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "n_high_conf=300" in rec
    assert str(MIN_N_HC_FOR_KEEP) in rec


def test_missing_metric_reverts():
    rec = recommendation(
        {"brier": None, "roi_units": None, "n_high_conf": 0},
        best_opt=BASELINE_0_2553,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "metric missing" in rec


def test_baseline_optimizer_extracts_first_baseline_row():
    rows = [
        {"status": "baseline", "opt_brier": "0.2553", "opt_roi": "54.55", "commit": "abc"},
        {"status": "kept", "opt_brier": "0.2420", "opt_roi": "124.64", "commit": "def"},
        {"status": "reverted", "opt_brier": "0.2500", "opt_roi": "90.00", "commit": "ghi"},
    ]
    b = baseline_optimizer(rows)
    assert b is not None
    assert b["brier"] == pytest.approx(0.2553)
    assert b["roi_units"] == pytest.approx(54.55)
    assert b["row_idx"] == 0


def test_baseline_optimizer_returns_none_when_no_baseline():
    rows = [{"status": "pending", "opt_brier": "0.25", "opt_roi": "0"}]
    assert baseline_optimizer(rows) is None


def test_baseline_optimizer_rejects_corrupt_baseline_row():
    rows = [{"status": "baseline", "opt_brier": "garbage", "opt_roi": "54.55", "commit": "x"}]
    with pytest.raises(SystemExit, match="Corrupt baseline row"):
        baseline_optimizer(rows)


def test_cmd_run_rejects_multiple_baseline_rows():
    # Adversarial review: cmd_run must enforce exactly-one-baseline before
    # the cumulative gate even runs, otherwise multiple baseline rows would
    # silently desync the cumulative anchor (first baseline) from the stop
    # condition anchor (most recent baseline/kept).
    import subprocess
    import tempfile
    from pathlib import Path
    import shutil

    REPO_ROOT = Path(__file__).resolve().parents[2]
    runner = REPO_ROOT / "mlb_research" / "run_experiment.py"
    real_tsv = REPO_ROOT / "mlb_research" / "results.tsv"

    # Read the real ledger header so we generate a same-shape fake.
    header = real_tsv.read_text().splitlines()[0]

    fake_row1 = "20260101T000000Z\taaaaaaa\t0.255300\t0.720000\t50.00\t0.5500\t800\t1400\t0.240000\t80.00\t480\t0.190000\t40.00\t78\tbaseline\tbaseline\tbaseline 1\tx\ty"
    fake_row2 = "20260102T000000Z\tbbbbbbb\t0.250000\t0.700000\t60.00\t0.5600\t820\t1400\t0.240000\t80.00\t480\t0.190000\t40.00\t78\tbaseline\tbaseline\tbaseline 2\tx\ty"

    # Use a backup-and-restore dance against the real results.tsv -- the
    # runner reads from a hardcoded path, so we have to swap the file.
    backup = real_tsv.read_bytes()
    try:
        real_tsv.write_text("\n".join([header, fake_row1, fake_row2]) + "\n")
        result = subprocess.run(
            ["python3", str(runner), "run",
             "--config", str(REPO_ROOT / "mlb_research" / "configs" / "baseline.json"),
             "--change-type", "features",
             "--description", "should be rejected -- two baselines"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
    finally:
        real_tsv.write_bytes(backup)

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Multiple baseline rows" in combined


def test_running_best_unchanged_by_new_code():
    # Sanity: running_best_optimizer still picks the lowest-Brier
    # baseline/kept row regardless of ordering.
    rows = [
        {"status": "baseline", "opt_brier": "0.2553", "opt_roi": "54.55", "commit": "a"},
        {"status": "kept", "opt_brier": "0.2420", "opt_roi": "124.64", "commit": "b"},
        {"status": "reverted", "opt_brier": "0.2200", "opt_roi": "150.00", "commit": "c"},
    ]
    b = running_best_optimizer(rows)
    assert b["brier"] == pytest.approx(0.2420)


def test_equal_to_running_best_reverts():
    # Marginal Δ = 0 exactly. Neither gate passes.
    rec = recommendation(
        _opt(RUN2_BEST_0_2420["brier"]),
        best_opt=RUN2_BEST_0_2420,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")


def test_constants_have_sensible_relationship():
    # Primary floor should be strictly stronger than cumulative marginal floor.
    assert MIN_BRIER_DELTA_FOR_KEEP > MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP
    # Cumulative floor should exceed primary (it's a longer path of wins).
    assert MIN_CUMULATIVE_DELTA_FOR_KEEP >= MIN_BRIER_DELTA_FOR_KEEP


def test_cumulative_gate_is_single_use_blocks_post_threshold_noise_stairstep():
    # Adversarial-review case: once running best is already past 0.015 from
    # baseline, the cumulative gate must NOT fire for sub-noise marginal wins.
    # Otherwise the ledger stair-steps forward on 0.001 nudges indefinitely.
    running_best_past_threshold = {"brier": 0.2402, "roi_units": 156.0}  # 0.0151 off baseline
    new = _opt(0.2392, roi=158.0, n_hc=1100)  # marginal 0.0010, cumulative 0.0161
    rec = recommendation(
        new,
        best_opt=running_best_past_threshold,
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("REVERT")
    assert "cumulative gate exhausted" in rec or "primary floor" in rec


def test_cumulative_gate_blocked_after_primary_keep_pushed_running_best_past_threshold():
    # If a primary-gate keep already lifted running best beyond 0.015 from
    # baseline (e.g. a strong family swap landing at 0.240), subsequent
    # cumulative-gate attempts must be blocked even with sub-noise marginal.
    running_best_via_primary = {"brier": 0.2400, "roi_units": 140.0}  # 0.0153 off baseline
    new = _opt(0.2395, roi=145.0, n_hc=1100)  # marginal 0.0005, cumulative 0.0158
    rec = recommendation(
        new, best_opt=running_best_via_primary, baseline_opt=BASELINE_0_2553, roi_regression_cap=3.0
    )
    assert rec.startswith("REVERT")


def test_primary_gate_still_works_after_cumulative_exhaustion():
    # Even after the cumulative gate is exhausted, a clean primary-floor
    # crossing should still KEEP (this is the bound on how much the single-use
    # rule restricts forward progress).
    running_best_past_threshold = {"brier": 0.2400, "roi_units": 140.0}
    new = _opt(0.2290, roi=180.0, n_hc=1100)  # marginal 0.0110, primary-clearing
    rec = recommendation(
        new, best_opt=running_best_past_threshold, baseline_opt=BASELINE_0_2553, roi_regression_cap=3.0
    )
    assert rec.startswith("KEEP")
    assert "primary gate" in rec


def test_cumulative_gate_fires_exactly_once_at_first_threshold_crossing():
    # The CANONICAL case: running best 0.2420 (0.0133 off baseline, NOT yet
    # crossed), candidate 0.2402 (cumulative 0.0151, marginal 0.0018). Must
    # KEEP via cumulative gate. Run 2 exp 21 ground truth.
    rec = recommendation(
        _opt(0.2402, roi=156.36, n_hc=1159),
        best_opt=RUN2_BEST_0_2420,  # 0.0133 from baseline -- gate available
        baseline_opt=BASELINE_0_2553,
        roi_regression_cap=3.0,
    )
    assert rec.startswith("KEEP")
    assert "cumulative gate" in rec
