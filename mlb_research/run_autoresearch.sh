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
# This is a thin wrapper. The research loop itself lives in program.md;
# everything the agent needs is there. This script just confirms state
# and launches Claude Code with a single prompt pointing at program.md.

set -euo pipefail

cd "$(dirname "$0")/.."

FROZEN_CSV="mlb_research/anchor/mlb_frozen.csv"
MANIFEST="mlb_research/anchor/anchor_manifest.json"
LEDGER="mlb_research/results.tsv"
PROGRAM="mlb_research/program.md"

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

BASELINE=$(head -2 "$LEDGER" | tail -1 | cut -f3)
echo "Preflight OK. Baseline opt_brier=${BASELINE}. Launching agent..."
echo

PROMPT="Read mlb_research/program.md and follow it exactly. Do not stop to ask whether you should continue. Execute experiments until a stop condition in program.md fires, then write RUN_SUMMARY.md."

if command -v claude >/dev/null 2>&1; then
    exec claude -p "$PROMPT"
else
    cat <<EOF
'claude' CLI not found on PATH. To launch manually, start a Claude Code
session in this directory and send this prompt:

  $PROMPT

Alternative: paste the prompt into an open Claude Code session.
EOF
fi
