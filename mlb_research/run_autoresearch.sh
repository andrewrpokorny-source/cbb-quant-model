#!/usr/bin/env bash
#
# Launch the MLB auto-research loop.
#
# Preconditions:
#   - The frozen anchor snapshot exists (mlb_research/anchor/mlb_frozen.csv).
#   - Baseline row exists in mlb_research/results.tsv.
#   - Git working tree is clean.
#
# Usage:
#   bash mlb_research/run_autoresearch.sh
#
# Side effects:
#   - Scrubs KALSHI_* and TELEGRAM_* credentials from the subprocess
#     environment so an experiment that accidentally imports kalshi/
#     or telegram_bot.py cannot hit live APIs.
#   - Tees agent stdout/stderr to mlb_research/agent.log for tail -f
#     visibility (gitignored).
#   - Marks any dangling `pending` rows in results.tsv as `superseded`
#     before launching, so a crash-recovered run doesn't keep a stale
#     half-experiment in the ledger.

set -euo pipefail

cd "$(dirname "$0")/.."

FROZEN_CSV="mlb_research/anchor/mlb_frozen.csv"
MANIFEST="mlb_research/anchor/anchor_manifest.json"
LEDGER="mlb_research/results.tsv"
PROGRAM="mlb_research/program.md"
LOGFILE="mlb_research/agent.log"

if [[ ! -f "$FROZEN_CSV" ]]; then
    echo "ERROR: frozen anchor not found. Run: uv run python mlb_research/anchor/snapshot_data.py"
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: anchor manifest not found."
    exit 1
fi
if [[ ! -f "$LEDGER" ]]; then
    echo "ERROR: results.tsv not found. Run the baseline experiment first."
    exit 1
fi
if [[ ! -f "$PROGRAM" ]]; then
    echo "ERROR: program.md not found."
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: working tree is not clean. Commit or stash before launching."
    git status --short
    exit 1
fi

# Sweep stale pending rows into `superseded` so running-best / stop-streak
# calculations don't get confused by leftovers from a prior crashed run.
uv run python - <<'PY'
import csv
import os

RESULTS = "mlb_research/results.tsv"
if not os.path.exists(RESULTS):
    raise SystemExit(0)

with open(RESULTS, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
if not rows:
    raise SystemExit(0)

changed = 0
for r in rows:
    if r.get("status") == "pending":
        r["status"] = "superseded"
        changed += 1

if changed:
    tmp = RESULTS + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, RESULTS)
    print(f"Swept {changed} dangling pending row(s) to status=superseded.")
PY

BASELINE=$(head -2 "$LEDGER" | tail -1 | cut -f3)
echo "Preflight OK. Baseline opt_brier=${BASELINE}."

# Scrub live-service credentials. The research loop must not touch Kalshi or
# Telegram even if the agent accidentally imports a module that uses them.
unset KALSHI_API_KEY KALSHI_PRIVATE_KEY_PATH KALSHI_KEY_ID
unset TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWED_USERS
export KALSHI_API_KEY="" KALSHI_PRIVATE_KEY_PATH="" TELEGRAM_BOT_TOKEN=""

echo "Credentials scrubbed: KALSHI_*, TELEGRAM_* blanked."
echo "Log streaming to ${LOGFILE}."
echo

PROMPT="Read mlb_research/program.md and follow it exactly. Do not stop to ask whether you should continue. Execute experiments until a stop condition in program.md fires (also enforced by run_experiment.py), then write mlb_research/RUN_SUMMARY.md."

if command -v claude >/dev/null 2>&1; then
    # Tee to logfile so the user can tail -f progress. Agent exit code is
    # preserved via pipefail.
    claude -p "$PROMPT" 2>&1 | tee -a "${LOGFILE}"
else
    cat <<EOF
'claude' CLI not found on PATH. To launch manually, start a Claude Code
session in this directory and send this prompt:

  $PROMPT

Alternative: paste the prompt into an open Claude Code session. Credentials
have been scrubbed from this shell but will be restored when you open a
new terminal.
EOF
fi
