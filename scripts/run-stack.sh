#!/usr/bin/env bash
# Launch the Streamlit dashboard and the Telegram bot together.
#
# Both services run as background process groups so the cleanup trap can
# signal the actual Python child (not just the `uv run` wrapper) via a
# process-group kill. Cleanup runs on any exit path -- SIGINT, SIGTERM, or
# the script exiting normally -- so neither service is ever orphaned.
#
# The bot is the lead process: when it exits, the script exits and the EXIT
# trap stops Streamlit. Streamlit output is appended to streamlit.log with a
# session header per run; the bot writes to telegram_bot.log via its own
# logging config.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STREAMLIT_LOG="$REPO_ROOT/streamlit.log"

# Preconditions: surface missing deps with an actionable message rather than
# letting them fall out as a deep import error inside the children.
command -v uv >/dev/null 2>&1 \
    || { echo "[run-stack] ERROR: 'uv' not found in PATH" >&2; exit 1; }
[[ -f app.py ]] \
    || { echo "[run-stack] ERROR: app.py not found in $REPO_ROOT" >&2; exit 1; }
[[ -f telegram_bot.py ]] \
    || { echo "[run-stack] ERROR: telegram_bot.py not found in $REPO_ROOT" >&2; exit 1; }

# Initialize empty so cleanup() is safe even if we die before launching.
STREAMLIT_PID=""
BOT_PID=""

# Stop a process group: SIGTERM, wait up to ~2.5s (5 x 0.5s), then SIGKILL.
# Targets the group (`-pgid`) rather than a single pid because `uv run` is a
# wrapper and signaling only its pid may not propagate to the actual child.
stop_group() {
    local pid="$1" label="$2"
    [[ -n "$pid" ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    echo "[run-stack] Stopping $label (pgid $pid)..."
    if ! kill -TERM -- "-$pid" 2>/dev/null; then
        echo "[run-stack] WARNING: failed to send SIGTERM to $label pgid $pid" >&2
    fi
    for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    if kill -0 "$pid" 2>/dev/null; then
        echo "[run-stack] WARNING: $label pgid $pid survived SIGKILL -- manual cleanup required" >&2
    fi
}

cleanup() {
    local code=$?
    # Disarm to avoid re-entry from a second signal during the kill window.
    trap - INT TERM EXIT
    stop_group "$BOT_PID" "Telegram bot"
    stop_group "$STREAMLIT_PID" "Streamlit"
    if [[ "$code" -ne 0 ]]; then
        echo "[run-stack] Exit code: $code"
    fi
    exit "$code"
}
trap cleanup INT TERM EXIT

# Mark this run in streamlit.log so a failure tail at the bottom of this
# script doesn't mix output from a previous session.
echo "===== [run-stack] $(date -Iseconds) pid=$$ =====" >>"$STREAMLIT_LOG"

# Job control puts each backgrounded command in its own process group with
# pgid == pid, which is what `kill -- -pgid` in stop_group depends on.
set -m

echo "[run-stack] Starting Streamlit (logs: $STREAMLIT_LOG)..."
uv run streamlit run app.py >>"$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!
echo "[run-stack] Streamlit pgid $STREAMLIT_PID"

# Wait up to ~1s (5 x 0.2s) so a fast Streamlit crash (import error, missing
# dep) surfaces before we hand off to the bot. This does NOT verify Streamlit
# bound port 8501 -- only that the process didn't die immediately.
for _ in 1 2 3 4 5; do
    sleep 0.2
    kill -0 "$STREAMLIT_PID" 2>/dev/null || break
done
if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    wait "$STREAMLIT_PID" 2>/dev/null
    streamlit_exit=$?
    echo "[run-stack] ERROR: Streamlit exited with code $streamlit_exit before bot could start." >&2
    echo "[run-stack] Last 20 lines of $STREAMLIT_LOG:" >&2
    tail -n 20 "$STREAMLIT_LOG" >&2 || echo "[run-stack] (could not read $STREAMLIT_LOG)" >&2
    exit 1
fi

echo "[run-stack] Starting Telegram bot in background (Ctrl-C to stop both)..."
uv run python telegram_bot.py &
BOT_PID=$!
echo "[run-stack] Telegram bot pgid $BOT_PID"

# Wait on the bot specifically: bare `wait` would return as soon as either
# child exits. When the bot exits (cleanly or otherwise), the EXIT trap
# stops Streamlit and the script propagates the bot's exit code.
wait "$BOT_PID"
