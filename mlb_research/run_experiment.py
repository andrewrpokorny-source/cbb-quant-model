"""Auto-research experiment runner and ledger.

Two subcommands:

    run       -- Score a config against the three-tier anchor and append a row
                 to results.tsv with status=pending. Archives the config and
                 the full metrics JSON under experiments/<timestamp>/.

    finalize  -- Mark the most recent pending row as kept / reverted / not-kept
                 / superseded and stamp the current git HEAD on it.

Workflow (from program.md):

    1. Edit your config, rebuild data, or patch code.
    2. `python run_experiment.py run --config cfg.json --change-type features \
            --description "dropped low-importance"`
    3. Read the "keep vs revert" recommendation.
    4. If keep: commit the change, then `run_experiment.py finalize --status kept`.
       If revert: `git restore .` (or revert), then `run_experiment.py finalize \
            --status reverted`.

The runner never makes keep/revert decisions itself. The agent reads the
optimizer columns (opt_brier, opt_roi) and applies the rule in program.md.
Monitor columns (mon25_*, mon26_*) are visible for human review but MUST NOT
drive decisions -- doing so defeats the overfitting guard.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ANCHOR_EVAL = os.path.join(RESEARCH_DIR, "anchor", "anchor_eval.py")
RESULTS_TSV = os.path.join(RESEARCH_DIR, "results.tsv")
EXPERIMENTS_DIR = os.path.join(RESEARCH_DIR, "experiments")

PER_EXPERIMENT_TIMEOUT_SECONDS = 600  # 10 min hard cap

# Stop conditions -- enforced in code, not just program.md.
MAX_EXPERIMENTS_SINCE_BASELINE = 50
MAX_CONSECUTIVE_NON_KEEPS = 15

# Keep-eligibility rules (applied by `recommendation`). At n=1403 games in the
# optimizer window, Brier standard error is ~0.007, so a single run's max-of-50
# Gaussian noise floor is ~ΔBrier ~= 0.015. We require a stricter 0.010 delta
# to be confident an improvement is not within-noise.
MIN_BRIER_DELTA_FOR_KEEP = 0.010
# Secondary "cumulative" gate: allows a row to be KEPT whose marginal delta
# vs the running best is sub-floor, provided the cumulative Brier drop vs the
# original baseline is clearly real and the marginal step is positive. Single-
# use by design: it fires ONLY when the running best has not yet crossed
# MIN_CUMULATIVE_DELTA_FOR_KEEP from baseline. Once any row (primary or
# secondary keep) puts the running best past that threshold, future
# experiments must clear the primary 0.010 marginal floor -- otherwise the
# ledger could stair-step forward on sub-noise 0.001 nudges indefinitely
# (caught by adversarial review pre-Run-3).
#
# The marginal minimum on the secondary gate is set low on purpose: the
# whole point of the gate is to pass sub-noise marginal wins (the canonical
# Run 2 case was Δ=+0.0018). It stays strictly positive to reject ties and
# pure-regression noise. The single-use semantics are what bounds the
# downside; the marginal floor inside the gate is just a tie-breaker.
MIN_CUMULATIVE_DELTA_FOR_KEEP = 0.015
MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP = 0.001
# Baseline produces ~795 high-confidence picks over the optimizer window. A
# genuine improvement should not gut the pick population by > ~35%. Otherwise
# a win may be "shrink toward 0.5" (Brier floor is 0.25 for uniform 0.5 output)
# which is not real alpha.
MIN_N_HC_FOR_KEEP = 500

LEDGER_COLUMNS = [
    "timestamp_utc",
    "commit",
    "opt_brier",
    "opt_log_loss",
    "opt_roi",
    "opt_high_conf_acc",
    "opt_n_hc",
    "opt_n_games",
    "mon25_brier",
    "mon25_roi",
    "mon25_n_hc",
    "mon26_brier",
    "mon26_roi",
    "mon26_n_hc",
    "status",
    "change_type",
    "description",
    "config_path",
    "archive_dir",
]

VALID_STATUSES = {"pending", "baseline", "kept", "reverted", "not-kept", "superseded"}
VALID_CHANGE_TYPES = {
    "baseline",
    "features",
    "hyperparams",
    "blend",
    "gate",
    "data",
    "training",
    "evaluation",
    "other",
}


def _write_tsv_atomic(rows: list[dict]):
    """Write the whole ledger via tmp-file + os.replace. A crash mid-write
    cannot leave a truncated results.tsv."""
    tmp = RESULTS_TSV + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LEDGER_COLUMNS})
    os.replace(tmp, RESULTS_TSV)


def append_row(row: dict):
    """Append one row. Reads-then-rewrites so the write is atomic."""
    rows = read_all_rows()
    rows.append(row)
    _write_tsv_atomic(rows)


def read_all_rows() -> list[dict]:
    if not os.path.exists(RESULTS_TSV):
        return []
    with open(RESULTS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rewrite_rows(rows: list[dict]):
    _write_tsv_atomic(rows)


def git_head_sha() -> str:
    """Return short HEAD SHA. Hard-fail if git is unavailable -- commit
    attribution is load-bearing for the ledger."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        sys.exit(f"Unable to resolve git HEAD SHA (required for attribution): {e}")
    sha = out.decode().strip()
    if not sha:
        sys.exit("git rev-parse returned empty SHA.")
    return sha


def run_eval(config_path: str | None, output_path: str) -> dict:
    """Run anchor_eval.py writing its JSON to `output_path`. Using a file
    instead of stdout avoids silent corruption from stray prints in the
    import chain (which would blow up json.loads)."""
    cmd = [sys.executable, ANCHOR_EVAL, "--output", output_path]
    if config_path:
        cmd += ["--model-config", config_path]
    try:
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=PER_EXPERIMENT_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        stderr = (e.stderr or "").strip()
        sys.exit(
            f"anchor_eval.py exceeded {PER_EXPERIMENT_TIMEOUT_SECONDS}s timeout."
            + (f"\nLast stderr: {stderr[-500:]}" if stderr else "")
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.exit(f"anchor_eval.py failed with exit code {e.returncode}.")

    if not os.path.exists(output_path):
        sys.exit(f"anchor_eval.py did not write {output_path}.")
    with open(output_path) as f:
        return json.load(f)


def fmt(v, digits=4):
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def recommendation(
    opt: dict,
    best_opt: dict | None,
    baseline_opt: dict | None,
    roi_regression_cap: float,
) -> str:
    """Recommend keep vs revert against the running-best optimizer row.

    KEEP requires n_high_conf >= MIN_N_HC_FOR_KEEP, non-None brier/roi, roi
    regression within cap, and ONE of:

    1. **Primary gate:** marginal brier delta vs running best >=
       MIN_BRIER_DELTA_FOR_KEEP (0.010).
    2. **Cumulative gate:** cumulative brier drop vs original baseline >=
       MIN_CUMULATIVE_DELTA_FOR_KEEP (0.015) AND marginal delta vs running
       best >= MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP (0.005). Unblocks
       stacked wins (e.g. prune + new family) where each single change is
       within-noise but the combination has moved meaningfully off baseline.
    """
    new_brier = opt.get("brier")
    new_roi = opt.get("roi_units")
    new_n_hc = opt.get("n_high_conf", 0)
    if new_brier is None or new_roi is None:
        return (
            "REVERT (optimizer metric missing -- "
            f"brier={new_brier}, roi={new_roi}). Likely zero high-conf "
            "picks or zero trained folds."
        )

    if new_n_hc < MIN_N_HC_FOR_KEEP:
        return (
            f"REVERT (n_high_conf={new_n_hc} below floor "
            f"{MIN_N_HC_FOR_KEEP}). A win with < {MIN_N_HC_FOR_KEEP} picks "
            "is likely resolution collapse, not alpha."
        )

    if best_opt is None:
        return "KEEP (no prior best -- this becomes the baseline)."

    best_brier = best_opt["brier"]
    best_roi = best_opt["roi_units"]
    delta = best_brier - new_brier

    roi_ok = new_roi > best_roi - roi_regression_cap
    brier_significant = delta >= MIN_BRIER_DELTA_FOR_KEEP

    if brier_significant and roi_ok:
        return (
            f"KEEP (primary gate: opt_brier {best_brier:.4f} -> {new_brier:.4f}, "
            f"Δ={delta:+.4f}; opt_roi {best_roi:+.2f}U -> {new_roi:+.2f}U; "
            f"n_hc={new_n_hc})."
        )

    if baseline_opt is not None and roi_ok:
        cumulative_delta = baseline_opt["brier"] - new_brier
        running_best_already_crossed = (
            (baseline_opt["brier"] - best_brier) >= MIN_CUMULATIVE_DELTA_FOR_KEEP
        )
        cumulative_ok = cumulative_delta >= MIN_CUMULATIVE_DELTA_FOR_KEEP
        marginal_ok = delta >= MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP
        if cumulative_ok and marginal_ok and not running_best_already_crossed:
            return (
                f"KEEP (cumulative gate: Δ_baseline={cumulative_delta:+.4f} "
                f"vs {baseline_opt['brier']:.4f} "
                f"(>={MIN_CUMULATIVE_DELTA_FOR_KEEP}); "
                f"marginal Δ={delta:+.4f} "
                f"(>={MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP}); "
                f"opt_roi {best_roi:+.2f}U -> {new_roi:+.2f}U; "
                f"n_hc={new_n_hc})."
            )

    reasons = []
    cumulative_gate_exhausted = (
        baseline_opt is not None
        and (baseline_opt["brier"] - best_brier) >= MIN_CUMULATIVE_DELTA_FOR_KEEP
    )
    if not brier_significant:
        if delta > 0:
            reasons.append(
                f"opt_brier improved by only {delta:.4f} (< primary floor "
                f"{MIN_BRIER_DELTA_FOR_KEEP})"
            )
        else:
            reasons.append(f"opt_brier did not improve ({best_brier:.4f} vs {new_brier:.4f})")
    if baseline_opt is not None and not cumulative_gate_exhausted:
        cum = baseline_opt["brier"] - new_brier
        if cum < MIN_CUMULATIVE_DELTA_FOR_KEEP:
            reasons.append(
                f"cumulative Δ vs baseline {cum:+.4f} (< secondary floor "
                f"{MIN_CUMULATIVE_DELTA_FOR_KEEP})"
            )
        elif delta < MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP:
            reasons.append(
                f"marginal Δ {delta:+.4f} below cumulative-gate minimum "
                f"{MIN_MARGINAL_DELTA_FOR_CUMULATIVE_KEEP}"
            )
    elif cumulative_gate_exhausted and not brier_significant and delta > 0:
        reasons.append(
            f"cumulative gate exhausted (running best already "
            f"{baseline_opt['brier'] - best_brier:+.4f} ahead of baseline; "
            f"primary floor applies)"
        )
    if not roi_ok:
        reasons.append(
            f"opt_roi regressed beyond cap {roi_regression_cap}U "
            f"({best_roi:+.2f}U -> {new_roi:+.2f}U)"
        )
    return "REVERT (" + "; ".join(reasons) + ")."


def running_best_optimizer(rows: list[dict]) -> dict | None:
    """Best opt_brier among rows with status in {baseline, kept}.

    Hard-fails if a baseline/kept row has an unparseable opt_brier or opt_roi:
    a corrupted comparator silently hides regressions and is the most
    dangerous form of ledger rot in an unattended run.
    """
    best = None
    for idx, r in enumerate(rows):
        if r.get("status") not in {"baseline", "kept"}:
            continue
        raw_brier = r.get("opt_brier", "")
        raw_roi = r.get("opt_roi", "")
        try:
            brier = float(raw_brier)
            roi = float(raw_roi)
        except (TypeError, ValueError) as e:
            sys.exit(
                f"Corrupt ledger row #{idx + 1} "
                f"(status={r.get('status')}, commit={r.get('commit')}): "
                f"cannot parse opt_brier={raw_brier!r} / opt_roi={raw_roi!r} ({e})"
            )
        if best is None or brier < best["brier"]:
            best = {"brier": brier, "roi_units": roi, "row_idx": idx}
    return best


def baseline_optimizer(rows: list[dict]) -> dict | None:
    """Return the single `status=baseline` row's optimizer metrics.

    The cumulative-delta secondary gate measures improvement against the
    ORIGINAL baseline, not the running best. Only the first baseline row is
    considered (the run_experiment invariant permits exactly one).
    """
    for idx, r in enumerate(rows):
        if r.get("status") != "baseline":
            continue
        raw_brier = r.get("opt_brier", "")
        raw_roi = r.get("opt_roi", "")
        try:
            return {
                "brier": float(raw_brier),
                "roi_units": float(raw_roi),
                "row_idx": idx,
            }
        except (TypeError, ValueError) as e:
            sys.exit(
                f"Corrupt baseline row #{idx + 1} "
                f"(commit={r.get('commit')}): cannot parse "
                f"opt_brier={raw_brier!r} / opt_roi={raw_roi!r} ({e})"
            )
    return None


def _enforce_stop_conditions(rows: list[dict]):
    """Exit cleanly if either hard cap has been reached.

    These are also stated in program.md but code enforcement exists so a
    confused agent cannot blow past them.

    Experiment count and revert-streak are computed relative to the most
    recent baseline/kept row, not all-time. This prevents pre-run artifacts
    (smoke tests, prior-session reverts) from seeding the stop condition.
    """
    # Find the index of the most recent baseline or kept row.
    anchor_idx = -1
    for i, r in enumerate(rows):
        if r.get("status") in {"baseline", "kept"}:
            anchor_idx = i

    experiments_since_anchor = rows[anchor_idx + 1:] if anchor_idx >= 0 else rows
    real_experiments = [
        r for r in experiments_since_anchor
        if r.get("status") not in {"baseline", "superseded"}
    ]

    if len(real_experiments) >= MAX_EXPERIMENTS_SINCE_BASELINE:
        sys.exit(
            f"STOP: experiment cap reached ({len(real_experiments)} experiments "
            f"since last baseline/kept, cap {MAX_EXPERIMENTS_SINCE_BASELINE}). "
            "Write mlb_research/RUN_SUMMARY.md and end the run."
        )

    # Trailing non-keeps counted from the most recent baseline/kept row
    # forward. Superseded rows (crash-recovery markers) don't break streaks.
    streak = 0
    for r in reversed(experiments_since_anchor):
        if r.get("status") in {"reverted", "not-kept", "pending"}:
            streak += 1
        elif r.get("status") == "superseded":
            continue  # session boundary, ignore
        else:
            break
    if streak >= MAX_CONSECUTIVE_NON_KEEPS:
        sys.exit(
            f"STOP: {streak} consecutive non-keeps since last baseline/kept "
            f"(cap {MAX_CONSECUTIVE_NON_KEEPS}). The hypothesis menu has "
            "likely plateaued. Write mlb_research/RUN_SUMMARY.md and end "
            "the run."
        )


def cmd_run(args):
    if args.change_type not in VALID_CHANGE_TYPES:
        sys.exit(f"--change-type must be one of {sorted(VALID_CHANGE_TYPES)}")

    config_path = args.config
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    # Running best is established BEFORE the eval so the recommendation can
    # be computed from a known-consistent snapshot of the ledger, and any
    # ledger corruption aborts before we spend compute on the eval.
    rows_before = read_all_rows()
    _enforce_stop_conditions(rows_before)

    # Enforce the baseline invariant. Every autonomous run must have exactly
    # one committed `status=baseline` row in the ledger before any pending
    # experiment can be recorded. The ONLY way to create that row is to
    # explicitly pass `--status baseline` as a one-time bootstrap. This
    # prevents silently "rebaselining" after a ledger wipe or manual edit.
    has_baseline = any(r.get("status") == "baseline" for r in rows_before)
    if args.status == "baseline":
        if has_baseline:
            sys.exit(
                "A baseline row already exists in results.tsv. Refusing to "
                "create a second one. If you want to re-baseline the whole "
                "run, reset the ledger deliberately (human action, not agent)."
            )
    else:
        if not has_baseline:
            sys.exit(
                "No baseline row found in results.tsv. Before running any "
                "experiments, bootstrap with:\n"
                "  uv run python mlb_research/run_experiment.py run "
                "--config mlb_research/configs/baseline.json --change-type "
                "baseline --description '...' --status baseline"
            )

    best = running_best_optimizer(rows_before)
    baseline = baseline_optimizer(rows_before)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_head_sha()
    archive_dir = os.path.join(EXPERIMENTS_DIR, f"{timestamp}_{commit}")
    os.makedirs(archive_dir, exist_ok=True)

    metrics_path = os.path.join(archive_dir, "metrics.json")
    results = run_eval(config_path, metrics_path)

    archived_config_path = os.path.join(archive_dir, "config.json")
    if config_path and os.path.exists(config_path):
        shutil.copyfile(config_path, archived_config_path)
    else:
        with open(archived_config_path, "w") as dst:
            json.dump({"_note": "anchor_eval defaults"}, dst, indent=2)
    with open(os.path.join(archive_dir, "description.txt"), "w") as f:
        f.write(args.description + "\n")

    # Validate shape before writing anything to results.tsv. A KeyError here
    # would leave the archive dir orphaned with no ledger row.
    for key in ("optimizer", "monitor_2025_tail", "monitor_2026"):
        if key not in results:
            sys.exit(f"anchor_eval output missing top-level key: {key}")

    opt = results["optimizer"]
    mon25 = results["monitor_2025_tail"]
    mon26 = results["monitor_2026"]

    row = {
        "timestamp_utc": timestamp,
        "commit": commit,
        "opt_brier": fmt(opt["brier"], 6),
        "opt_log_loss": fmt(opt["log_loss"], 6),
        "opt_roi": fmt(opt["roi_units"], 2),
        "opt_high_conf_acc": fmt(opt["high_conf_accuracy"], 4),
        "opt_n_hc": opt["n_high_conf"],
        "opt_n_games": opt["n_games"],
        "mon25_brier": fmt(mon25["brier"], 6),
        "mon25_roi": fmt(mon25["roi_units"], 2),
        "mon25_n_hc": mon25["n_high_conf"],
        "mon26_brier": fmt(mon26["brier"], 6),
        "mon26_roi": fmt(mon26["roi_units"], 2),
        "mon26_n_hc": mon26["n_high_conf"],
        "status": args.status,
        "change_type": args.change_type,
        "description": args.description,
        "config_path": os.path.relpath(archived_config_path, REPO_ROOT),
        "archive_dir": os.path.relpath(archive_dir, REPO_ROOT),
    }
    append_row(row)

    rec = recommendation(opt, best, baseline, args.roi_regression_cap)

    # Monitor columns are deliberately NOT printed. They exist in
    # results.tsv for human review but should not influence the agent's
    # next hypothesis. Monitors are also archived in metrics.json.
    print("=" * 72)
    print(f"EXPERIMENT: {args.description}")
    print(f"  change_type={args.change_type}  status={args.status}")
    print(f"  archive={row['archive_dir']}")
    print()
    print(
        f"  OPTIMIZER  brier={fmt(opt['brier'])}  "
        f"roi={fmt(opt['roi_units'], 2)}U  "
        f"n_hc={opt['n_high_conf']}  n_games={opt['n_games']}"
    )
    diag = results.get("_meta", {}).get("diagnostics", {}).get("optimizer", {})
    if diag:
        print(
            f"  (folds={diag.get('n_folds_trained')}, "
            f"skipped={diag.get('n_folds_skipped_thin_train')}+"
            f"{diag.get('n_folds_skipped_empty_week')}, "
            f"train_rows min/mean/max="
            f"{diag.get('train_rows_min')}/"
            f"{int(diag['train_rows_mean']) if diag.get('train_rows_mean') else None}/"
            f"{diag.get('train_rows_max')})"
        )
    print()
    print(f"  Recommendation: {rec}")
    print("=" * 72)


FINALIZE_ALLOWED_STATUSES = VALID_STATUSES - {"pending", "baseline"}


def cmd_finalize(args):
    if args.status not in FINALIZE_ALLOWED_STATUSES:
        sys.exit(
            f"--status must be one of {sorted(FINALIZE_ALLOWED_STATUSES)}. "
            "'baseline' can only be set via the 'run' subcommand's --status flag "
            "to prevent a second baseline from bypassing the single-baseline invariant."
        )

    rows = read_all_rows()
    if not rows:
        sys.exit("results.tsv is empty; nothing to finalize.")

    # Find most recent pending row.
    idx = None
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get("status") == "pending":
            idx = i
            break
    if idx is None:
        sys.exit("No pending row found in results.tsv.")

    # Always stamp HEAD. Allowing an explicit --commit override was an
    # attribution-lie vector: an agent could finalize with any SHA string
    # it liked. HEAD at finalize time is the source of truth.
    rows[idx]["status"] = args.status
    rows[idx]["commit"] = git_head_sha()

    rewrite_rows(rows)
    print(f"Finalized row {idx + 1}: status={args.status} commit={rows[idx]['commit']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Score a config and append to results.tsv.")
    p_run.add_argument("--config", default=None, help="Path to model config JSON.")
    p_run.add_argument(
        "--change-type",
        required=True,
        help=f"One of: {sorted(VALID_CHANGE_TYPES)}",
    )
    p_run.add_argument("--description", required=True, help="Short experiment description.")
    p_run.add_argument(
        "--status",
        default="pending",
        help="Initial status. Usually 'pending' (finalize later) or 'baseline'.",
    )
    p_run.add_argument(
        "--roi-regression-cap",
        type=float,
        default=3.0,
        help="Max optimizer ROI regression (units) allowed when Brier improves.",
    )
    p_run.set_defaults(func=cmd_run)

    p_fin = sub.add_parser("finalize", help="Update most recent pending row.")
    p_fin.add_argument(
        "--status",
        required=True,
        help="kept | reverted | not-kept | superseded | baseline",
    )
    p_fin.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
