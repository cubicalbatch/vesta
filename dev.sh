#!/usr/bin/env bash
# Full dev mode: FastAPI backend (reload on src/vesta changes) + Vite SPA (HMR).
#
# Vite proxies /api, /health and /dev to the backend, so the backend MUST be
# serving before Vite starts — otherwise the first page load 502s (the original
# `make dev` race). This script starts the backend, blocks until /health is up,
# then launches the frontend, and tears both down cleanly on Ctrl-C or crash.
set -euo pipefail

 PORT="${VESTA_BACKEND_PORT:-5586}"
 PROXY="http://127.0.0.1:${PORT}"

# Surface the Settings → Advanced tab (eval/benchmarks) in `make dev`. Off by
# default in other environments; gated by GET /health's `advanced_menu` flag.
export VESTA_ADVANCED_MENU="${VESTA_ADVANCED_MENU:-True}"

# Send a signal to the whole process group on exit/Ctrl-C. Putting the trap on
# EXIT (not just INT/TERM) means a backend crash -> `wait -n` returns -> cleanup
# also takes down Vite, instead of leaving it dangling.
cleanup() {
	trap - EXIT INT TERM
	kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1) Backend: uvicorn --reload, port freed first by start.sh. Backgrounded so we
#    can gate the frontend on its readiness.
./start.sh &
BACKEND_PID=$!

# 2) Wait for the backend to actually serve before starting Vite. Startup can be
#    slow: the lifespan probes the answer model, which times out for *minutes*
#    when the inference gateway is unreachable (observed ~6 min here). So the
#    ceiling is generous and env-tunable; set VESTA_BACKEND_STARTUP_TIMEOUT to
#    shrink it when the gateway is fast. Bail early if the backend died (import
#    error, port clash start.sh couldn't resolve, …) so we never hang for nothing.
STARTUP_TIMEOUT="${VESTA_BACKEND_STARTUP_TIMEOUT:-600}" # seconds (default 10 min)
STEPS=$(( STARTUP_TIMEOUT * 2 )) # 0.5s poll interval
echo "dev.sh: waiting for backend at ${PROXY}/health (up to ${STARTUP_TIMEOUT}s) ..."
for i in $(seq 1 "$STEPS"); do
	if curl -sf "${PROXY}/health" >/dev/null 2>&1; then
		echo "dev.sh: backend ready"
		break
	fi
	if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
		echo "dev.sh: backend exited before becoming ready" >&2
		exit 1
	fi
	if [ $(( i % 30 )) -eq 0 ]; then
		echo "dev.sh:   …still waiting ($(( i / 2 ))s)"
	fi
	sleep 0.5
	if [ "$i" -eq "$STEPS" ]; then
		echo "dev.sh: backend did not become healthy within ${STARTUP_TIMEOUT}s" >&2
		exit 1
	fi
done

# 3) Frontend: Vite dev server with HMR, proxying API calls to the backend.
cd frontend
VESTA_API_PROXY_TARGET="$PROXY" npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

# 4) Block until EITHER process exits, then let the EXIT trap clean up the other.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
