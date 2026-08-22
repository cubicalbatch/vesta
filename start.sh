#!/usr/bin/env bash
# Start Vesta in dev mode: uvicorn with hot reload on port 5586.
set -euo pipefail

PORT="${VESTA_BACKEND_PORT:-5586}"

# If something is already listening on the port, kill it first so uvicorn can bind.
if pid=$(lsof -ti tcp:"$PORT"); then
    echo "start.sh: port $PORT in use by pid $pid; killing it" >&2
    kill $pid 2>/dev/null || true
    # Wait (up to ~5s) for the port to actually free up.
    for _ in $(seq 1 50); do
        lsof -ti tcp:"$PORT" >/dev/null 2>&1 || break
        sleep 0.1
    done
    if lsof -ti tcp:"$PORT" >/dev/null 2>&1; then
        echo "start.sh: process on port $PORT did not exit; sending SIGKILL" >&2
        kill -9 $pid 2>/dev/null || true
    fi
fi

# Hot reload — but scoped to backend source only. The default watcher sees the
# whole CWD, so the ever-changing SQLite -wal/-shm files under data/, index logs,
# and Vite's .svelte-kit churn would trigger a constant reload storm (and are why
# --reload was previously disabled). Restricting to src/vesta keeps reloads to
# actual backend code changes.
exec uv run uvicorn vesta.main:app \
	--host 0.0.0.0 --port "$PORT" \
	--reload --reload-dir src/vesta
