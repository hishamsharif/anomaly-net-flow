#!/usr/bin/env bash
# =============================================================================
# AnomalyNetFlow — Operations
#
# Usage:  bash ops.sh <command>
#
# Commands:
#   start               Start the API (training runs inside it as threads)
#   stop                Stop the API
#   restart             Restart the API
#   status              Show whether the API is running + health check
#   logs                Tail the API log (Ctrl-C to exit)
#   ui                  Start the React UI dev server  (port 5173)
#   train [mode]        Trigger training — mode: initial (default) | fine_tune
#   open                Open the UI in your default browser
#
# Run setup.sh first if you haven't already.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="dl_env"
PID_DIR="$SCRIPT_DIR/logs"
API_PID="$PID_DIR/api.pid"
API_LOG="$PID_DIR/api.log"
API_PORT=8000
UI_PORT=5173

mkdir -p "$PID_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "  ${GREEN}✔${NC}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
fail() { echo -e "  ${RED}✘${NC}  $*"; exit 1; }
info() { echo -e "  ${CYAN}→${NC}  $*"; }
sep()  { echo -e "${CYAN}──────────────────────────────────────────────────────────${NC}"; }

# ── Helpers ───────────────────────────────────────────────────────────────────
_is_running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

_free_port() {
  local pid; pid=$(lsof -ti :"$1" 2>/dev/null || true)
  [ -n "$pid" ] && { warn "Port $1 in use — stopping old process (PID $pid)"; kill "$pid" 2>/dev/null || true; sleep 1; }
}

_verify_setup() {
  local issues=0

  # Conda env
  if ! conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}\s"; then
    warn "Conda env '${CONDA_ENV}' not found"
    issues=$((issues + 1))
  fi

  # Python packages (check fastapi as a proxy for requirements.txt)
  if ! conda run -n "$CONDA_ENV" python -c "import fastapi" 2>/dev/null; then
    warn "Python dependencies not installed (fastapi missing)"
    issues=$((issues + 1))
  fi

  # UI node_modules
  if [ ! -d "$SCRIPT_DIR/ui/node_modules" ]; then
    warn "UI dependencies not installed (ui/node_modules missing)"
    issues=$((issues + 1))
  fi

  # .env
  if [ ! -f "$SCRIPT_DIR/.env" ] && [ ! -f "$SCRIPT_DIR/.env.example" ]; then
    warn ".env.example missing — repository may be incomplete"
    issues=$((issues + 1))
  fi

  # Dataset (warn only — training will fail but the API starts fine)
  local csv_count
  csv_count=$(find -L "$SCRIPT_DIR/data/CIC-IDS2017" -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
  if [ "$csv_count" -eq 0 ]; then
    warn "No CIC-IDS2017 CSV files found in data/CIC-IDS2017/ — initial training will fail"
  fi

  if [ "$issues" -gt 0 ]; then
    echo ""
    fail "$issues setup issue(s) found. Run:  bash setup.sh"
  fi
}

_check_setup() {
  if ! conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}\s"; then
    fail "Conda env '${CONDA_ENV}' not found. Run:  bash setup.sh"
  fi
}

_open_url() {
  local url="$1"
  if command -v xdg-open &>/dev/null; then xdg-open "$url"
  elif command -v open &>/dev/null; then open "$url"
  else info "Open manually: $url"; fi
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_start() {
  _verify_setup
  if _is_running "$API_PID"; then
    ok "API already running (PID $(cat "$API_PID")) — http://localhost:${API_PORT}"
    return
  fi
  _free_port "$API_PORT"
  info "Starting API on port ${API_PORT} ..."
  PYTHONPATH="$SCRIPT_DIR" conda run -n "$CONDA_ENV" \
    uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" >> "$API_LOG" 2>&1 &
  echo $! > "$API_PID"
  sleep 4
  if _is_running "$API_PID"; then
    sep
    ok "API running  →  http://localhost:${API_PORT}/docs"
    ok "Log file     →  $API_LOG"
    sep
    echo ""
    info "Start the UI in a separate terminal:  bash ops.sh ui"
    echo ""
  else
    fail "API failed to start — check the log: $API_LOG"
  fi
}

cmd_stop() {
  if _is_running "$API_PID"; then
    kill "$(cat "$API_PID")" 2>/dev/null || true
    rm -f "$API_PID"
  fi
  local pid; pid=$(lsof -ti :"$API_PORT" 2>/dev/null || true)
  [ -n "$pid" ] && { kill "$pid" 2>/dev/null || true; sleep 1; }
  lsof -i :"$API_PORT" &>/dev/null && warn "Port ${API_PORT} still in use" || ok "API stopped"
}

cmd_restart() {
  info "Restarting API ..."
  cmd_stop
  sleep 1
  cmd_start
}

cmd_status() {
  sep
  echo -e "  ${BOLD}AnomalyNetFlow — Status${NC}"
  sep
  if _is_running "$API_PID"; then
    printf "  API      ${GREEN}running${NC}   PID $(cat "$API_PID")   http://localhost:${API_PORT}\n"
    if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
      printf "  Health   ${GREEN}OK${NC}\n"
    else
      printf "  Health   ${YELLOW}not responding${NC}\n"
    fi
  else
    printf "  API      ${YELLOW}stopped${NC}\n"
    printf "  Run:     bash ops.sh start\n"
  fi
  sep
  echo ""
}

cmd_logs() {
  if [ ! -f "$API_LOG" ]; then
    warn "No log file yet. Start the API first: bash ops.sh start"
    exit 0
  fi
  info "Tailing $API_LOG  (Ctrl-C to stop)"
  tail -f "$API_LOG"
}

cmd_ui() {
  _check_setup
  if ! command -v node &>/dev/null; then
    fail "Node.js not found. Run bash setup.sh first."
  fi
  sep
  ok "Starting UI dev server on port ${UI_PORT}"
  ok "URL  →  http://localhost:${UI_PORT}"
  info "API is proxied automatically — make sure the API is running in another terminal"
  sep
  echo ""
  cd "$SCRIPT_DIR/ui" && npm run dev
}

cmd_train() {
  local mode="${1:-initial}"
  if [[ "$mode" != "initial" && "$mode" != "fine_tune" ]]; then
    fail "Unknown mode '$mode'. Use: initial | fine_tune"
  fi
  if ! curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
    fail "API is not running. Start it first:  bash ops.sh start"
  fi
  info "Submitting training job (mode: ${mode}) ..."
  resp=$(curl -s -X POST "http://localhost:${API_PORT}/api/training/start" \
    -H "Content-Type: application/json" \
    -d "{\"mode\":\"${mode}\"}")
  run_id=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('run_id','ERROR'))" 2>/dev/null || echo "ERROR")
  if [ "$run_id" = "ERROR" ]; then
    fail "Failed to start training. Response: $resp"
  fi
  sep
  ok "Training started  —  mode: ${mode}"
  ok "Run ID: ${run_id}"
  info "Stream live progress:"
  echo ""
  echo "    curl -N http://localhost:${API_PORT}/api/training/events/${run_id}"
  echo ""
  info "Or watch the Training tab in the UI: http://localhost:${UI_PORT}"
  sep
}

cmd_open() {
  if ! _is_running "$API_PID"; then
    warn "API is not running. Starting it first ..."
    cmd_start
  fi
  _open_url "http://localhost:${UI_PORT}"
}

cmd_help() {
  sep
  echo -e "  ${BOLD}AnomalyNetFlow — ops.sh${NC}"
  sep
  echo ""
  echo -e "  ${BOLD}Usage:${NC}  bash ops.sh <command>"
  echo ""
  echo -e "  ${CYAN}start${NC}              Start the API server"
  echo -e "  ${CYAN}stop${NC}               Stop the API server"
  echo -e "  ${CYAN}restart${NC}            Restart the API server"
  echo -e "  ${CYAN}status${NC}             Show running status + health check"
  echo -e "  ${CYAN}logs${NC}               Tail the API log (Ctrl-C to exit)"
  echo -e "  ${CYAN}ui${NC}                 Start the React UI  (port ${UI_PORT})"
  echo -e "  ${CYAN}train [mode]${NC}       Start training — mode: initial (default) | fine_tune"
  echo -e "  ${CYAN}open${NC}               Open the UI in your browser"
  echo -e "  ${CYAN}verify${NC}             Check setup is complete and ready to run"
  echo ""
  echo -e "  ${BOLD}First time?${NC}  Run  ${CYAN}bash setup.sh${NC}  first."
  echo ""
  sep
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  ui)      cmd_ui ;;
  train)   cmd_train "${1:-initial}" ;;
  open)    cmd_open ;;
  verify)  _verify_setup && ok "Setup verified — ready to run." ;;
  help|--help|-h) cmd_help ;;
  *) fail "Unknown command: '$COMMAND'  —  run: bash ops.sh help" ;;
esac
