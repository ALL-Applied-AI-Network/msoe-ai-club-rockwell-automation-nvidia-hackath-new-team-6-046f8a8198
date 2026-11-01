#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
CONTAINER="${ISAAC_CONTAINER:-isaac}"
CONTAINER_ROOT="${CONTAINER_WORKSPACE_ROOT:-/workspace}"
VIAL_ROOT="$CONTAINER_ROOT/vial-inspection"
SERVICE="$VIAL_ROOT/sdg/inspection_service.py"
LABELER="$VIAL_ROOT/vial-labeler-participants-20261014/vial-labeler-participants-v1/labeler.py"
BUNDLE="$VIAL_ROOT/sdg/output/deployment-bundle-v1"
INCOMING="$VIAL_ROOT/live/incoming"
RESULTS="$VIAL_ROOT/live/results"
LOG="$VIAL_ROOT/live/inspection-service.log"
REPORT_LOG="$VIAL_ROOT/live/report-server.log"
REPORT_PORT="${REPORT_PORT:-8765}"

container_exec() {
  docker exec "$CONTAINER" "$@"
}

validate() {
  container_exec test -f "$SERVICE"
  container_exec test -f "$LABELER"
  container_exec test -f "$BUNDLE/manifest.json"
  container_exec mkdir -p "$INCOMING" "$RESULTS"
}

service_running() {
  container_exec bash -lc "pgrep -f '[i]nspection_service.py' >/dev/null"
}

report_running() {
  container_exec bash -lc "pgrep -f '[h]ttp.server $REPORT_PORT .*live/results' >/dev/null"
}

start_runtime() {
  validate
  if ! service_running; then
    docker exec -d "$CONTAINER" bash -lc \
      "exec /isaac-sim/python.sh '$SERVICE' --labeler '$LABELER' --bundle '$BUNDLE' --input '$INCOMING' --output '$RESULTS' --confirm-sessions --minimum-session-frames 3 --poll-seconds 1 >>'$LOG' 2>&1"
  fi
  if ! report_running; then
    docker exec -d "$CONTAINER" bash -lc \
      "exec /isaac-sim/python.sh -m http.server '$REPORT_PORT' --bind 127.0.0.1 --directory '$RESULTS' >>'$REPORT_LOG' 2>&1"
  fi
  echo "inspection service: running"
  echo "report server: http://127.0.0.1:$REPORT_PORT/"
}

stop_runtime() {
  container_exec bash -lc "pkill -TERM -f '[i]nspection_service.py' || true"
  container_exec bash -lc "pkill -TERM -f '[h]ttp.server $REPORT_PORT .*live/results' || true"
  echo "inspection runtime: stopped"
}

show_status() {
  if service_running; then
    echo "inspection service: running"
  else
    echo "inspection service: stopped"
  fi
  if report_running; then
    echo "report server: running on 127.0.0.1:$REPORT_PORT"
  else
    echo "report server: stopped"
  fi
  container_exec bash -lc "test -f '$RESULTS/health.json' && cat '$RESULTS/health.json' || true"
}

run_once() {
  validate
  container_exec /isaac-sim/python.sh "$SERVICE" \
    --labeler "$LABELER" --bundle "$BUNDLE" --input "$INCOMING" \
    --output "$RESULTS" --confirm-sessions --minimum-session-frames 3 --once
}

case "$ACTION" in
  start) start_runtime ;;
  stop) stop_runtime ;;
  restart) stop_runtime; start_runtime ;;
  status) show_status ;;
  once) run_once ;;
  *) echo "usage: $0 {start|stop|restart|status|once}" >&2; exit 2 ;;
esac
