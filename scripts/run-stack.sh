#!/usr/bin/env bash
# Launch the Streamlit dashboard and the Telegram bot together.
#
# Streamlit runs in the background; its logs go to streamlit.log.
# The Telegram bot runs in the foreground so its output stays visible.
# Ctrl-C (SIGINT) or a TERM signal stops both cleanly.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STREAMLIT_LOG="$REPO_ROOT/streamlit.log"

# Track the Streamlit child so we can clean it up on exit.
STREAMLIT_PID=""

cleanup() {
    local code=$?
    if [[ -n "$STREAMLIT_PID" ]] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo
        echo "[run-stack] Stopping Streamlit (pid $STREAMLIT_PID)..."
        kill "$STREAMLIT_PID" 2>/dev/null || true
        # Give it a moment to shut down, then force if needed.
        for _ in 1 2 3 4 5; do
            kill -0 "$STREAMLIT_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "$STREAMLIT_PID" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup INT TERM EXIT

echo "[run-stack] Starting Streamlit (logs: $STREAMLIT_LOG)..."
uv run streamlit run app.py >>"$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!
echo "[run-stack] Streamlit pid $STREAMLIT_PID"

# Brief pause so an immediate Streamlit crash surfaces before the bot starts.
sleep 1
if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    echo "[run-stack] Streamlit failed to start. Check $STREAMLIT_LOG"
    exit 1
fi

echo "[run-stack] Starting Telegram bot in foreground (Ctrl-C to stop both)..."
uv run python telegram_bot.py
