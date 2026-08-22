#!/usr/bin/env bash
#
# kiwi-start.sh — serve every ZIM in data/zims/ with kiwix-serve on a random
# port in the 20k range (20000–29999).
#
# Usage:
#   ./kiwi-start.sh            start (or restart) on a random 20k port
#   ./kiwi-start.sh 23456      start on a specific port
#   ./kiwi-start.sh stop       stop the running instance
#
# Override port via $KIWIX_PORT if you prefer. The server is backgrounded;
# logs go to /tmp/kiwix-serve.log and the PID is tracked in /tmp/kiwix-serve.pid
# so re-running stops the previous instance first.
#
# NOTE: a ZIM that is mid-download / incomplete cannot be added to kiwix's
# library and causes kiwix-serve to exit. Just re-run once the download is done.

set -euo pipefail

cd "$(dirname "$0")"

PIDFILE="/tmp/kiwix-serve.pid"
LOGFILE="/tmp/kiwix-serve.log"

stop() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping kiwix-serve (pid $pid)"
      kill "$pid"
      for _ in $(seq 1 25); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
}

if [[ "${1:-}" == "stop" ]]; then
  stop
  echo "Stopped."
  exit 0
fi

stop

# Port selection: explicit arg > $KIWIX_PORT > random free port in the 20k range.
if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ ]]; then
  PORT="$1"
elif [[ -n "${KIWIX_PORT:-}" ]]; then
  PORT="$KIWIX_PORT"
else
  PORT=$((RANDOM % 10000 + 20000))
  while ss -ltn 2>/dev/null | grep -q ":$PORT "; do
    PORT=$((RANDOM % 10000 + 20000))
  done
fi

shopt -s nullglob
ZIMS=(data/zims/*.zim)
shopt -u nullglob

if (( ${#ZIMS[@]} == 0 )); then
  echo "No ZIM files found in data/zims/" >&2
  exit 1
fi

echo "Serving ${#ZIMS[@]} ZIM(s):"
for z in "${ZIMS[@]}"; do
  echo "  - $z"
done

nohup kiwix-serve --port "$PORT" --threads 4 "${ZIMS[@]}" > "$LOGFILE" 2>&1 &
pid=$!
echo "$pid" > "$PIDFILE"

echo
echo "kiwix-serve started (pid $pid)"
echo "Browse: http://127.0.0.1:$PORT"
echo "Log:    $LOGFILE   (stop with: $0 stop)"
