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
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
ANCHOR_EVAL = os.path.join(RESEARCH_DIR, "anchor", "anchor_eval.py")
RESULTS_TSV = os.path.join(RESEARCH_DIR, "results.tsv")
EXPERIMENTS_DIR = os.path.join(RESEARCH_DIR, "experiments")

PER_EXPERIMENT_TIMEOUT_SECONDS = 600  # 10 min hard cap

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


def ensure_ledger_exists():
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(LEDGER_COLUMNS)


def append_row(row: dict):
    ensure_ledger_exists()
    with open(RESULTS_TSV, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([row.get(c, "") for c in LEDGER_COLUMNS])


def read_all_rows() -> list[dict]:
    if not os.path.exists(RESULTS_TSV):
        return []
    with open(RESULTS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rewrite_rows(rows: list[dict]):
    with open(RESULTS_TSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LEDGER_COLUMNS})


def git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return ""


def run_eval(config_path: str | None) -> dict:
    cmd = [sys.executable, ANCHOR_EVAL]
    if config_path:
        cmd += ["--model-config", config_path]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=PER_EXPERIMENT_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.TimeoutExpired:
        sys.exit(f"anchor_eval.py exceeded {PER_EXPERIMENT_TIMEOUT_SECONDS}s timeout.")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr)
        sys.exit(f"anchor_eval.py failed with exit code {e.returncode}.")
    return json.loads(proc.stdout)


def fmt(v, digits=4):
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def recommendation(opt: dict, best_opt: dict | None, roi_regression_cap: float) -> str:
    """Recommend keep vs revert against the running-best optimizer row."""
    if best_opt is None:
        return "KEEP (no prior best -- this becomes the baseline)."
    try:
        new_brier = float(opt["brier"])
        new_roi = float(opt["roi_units"])
        best_brier = float(best_opt["brier"])
        best_roi = float(best_opt["roi_units"])
    except (TypeError, ValueError, KeyError):
        return "UNDECIDED (missing metrics)."

    brier_better = new_brier < best_brier
    roi_ok = new_roi > best_roi - roi_regression_cap

    if brier_better and roi_ok:
        return (
            f"KEEP (opt_brier {best_brier:.4f} -> {new_brier:.4f}, "
            f"opt_roi {best_roi:+.2f}U -> {new_roi:+.2f}U)."
        )
    reasons = []
    if not brier_better:
        reasons.append(f"opt_brier did not improve ({best_brier:.4f} vs {new_brier:.4f})")
    if not roi_ok:
        reasons.append(
            f"opt_roi regressed beyond cap {roi_regression_cap}U "
            f"({best_roi:+.2f}U -> {new_roi:+.2f}U)"
        )
    return "REVERT (" + "; ".join(reasons) + ")."


def running_best_optimizer(rows: list[dict]) -> dict | None:
    """Best opt_brier among rows with status in {baseline, kept}."""
    best = None
    for r in rows:
        if r.get("status") not in {"baseline", "kept"}:
            continue
        try:
            brier = float(r["opt_brier"])
        except (TypeError, ValueError, KeyError):
            continue
        if best is None or brier < float(best["opt_brier"]):
            best = r
    if best is None:
        return None
    return {"brier": best["opt_brier"], "roi_units": best["opt_roi"]}


def cmd_run(args):
    if args.change_type not in VALID_CHANGE_TYPES:
        sys.exit(f"--change-type must be one of {sorted(VALID_CHANGE_TYPES)}")

    config_path = args.config
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    results = run_eval(config_path)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_head_sha()
    archive_dir = os.path.join(EXPERIMENTS_DIR, f"{timestamp}_{commit or 'no-commit'}")
    os.makedirs(archive_dir, exist_ok=True)

    archived_config = ""
    if config_path and os.path.exists(config_path):
        archived_config = os.path.join(archive_dir, "config.json")
        with open(config_path) as src, open(archived_config, "w") as dst:
            dst.write(src.read())
    else:
        # No config given: record that default was used.
        with open(os.path.join(archive_dir, "config.json"), "w") as dst:
            json.dump({"_note": "anchor_eval defaults"}, dst, indent=2)
        archived_config = os.path.join(archive_dir, "config.json")

    with open(os.path.join(archive_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(archive_dir, "description.txt"), "w") as f:
        f.write(args.description + "\n")

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
        "config_path": archived_config,
        "archive_dir": os.path.relpath(archive_dir, REPO_ROOT),
    }
    append_row(row)

    # Summary + recommendation (optimizer columns only).
    existing_rows = read_all_rows()[:-1]  # exclude the row we just wrote
    best = running_best_optimizer(existing_rows)
    rec = recommendation(opt, best, args.roi_regression_cap)

    print("=" * 72)
    print(f"EXPERIMENT: {args.description}")
    print(f"  change_type={args.change_type}  status={args.status}")
    print(f"  archive={row['archive_dir']}")
    print()
    print(f"  OPTIMIZER    brier={fmt(opt['brier'])} roi={fmt(opt['roi_units'],2):>7}U  n_hc={opt['n_high_conf']}")
    print(f"  (monitors, for human review -- do NOT drive decisions)")
    print(f"  mon_2025_tail brier={fmt(mon25['brier'])} roi={fmt(mon25['roi_units'],2):>7}U  n_hc={mon25['n_high_conf']}")
    print(f"  mon_2026      brier={fmt(mon26['brier'])} roi={fmt(mon26['roi_units'],2):>7}U  n_hc={mon26['n_high_conf']}")
    print()
    print(f"  Recommendation: {rec}")
    print("=" * 72)


def cmd_finalize(args):
    if args.status not in VALID_STATUSES or args.status == "pending":
        sys.exit(f"--status must be one of {sorted(VALID_STATUSES - {'pending'})}")

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

    rows[idx]["status"] = args.status
    if args.commit:
        rows[idx]["commit"] = args.commit
    else:
        sha = git_head_sha()
        if sha:
            rows[idx]["commit"] = sha

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
    p_fin.add_argument("--commit", default=None, help="Git SHA (defaults to HEAD).")
    p_fin.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
