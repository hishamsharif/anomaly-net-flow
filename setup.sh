#!/usr/bin/env bash
# =============================================================================
# AnomalyNetFlow — Setup
#
# Default (Docker — recommended for evaluation):
#   ./setup.sh
#
# Local / conda fallback (no Docker required):
#   ./setup.sh --local
#
# Prerequisites for Docker path:
#   - Docker Desktop  https://docs.docker.com/get-docker/
#
# Prerequisites for --local path:
#   - Miniconda or Anaconda  https://docs.conda.io/en/latest/miniconda.html
#   - Node.js >= 20          https://nodejs.org
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

step()  { echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC}  $*"; }
warn()  { echo -e "  ${YELLOW}⚠${NC}  $*"; }
fail()  { echo -e "  ${RED}✘${NC}  $*"; exit 1; }
info()  { echo -e "     $*"; }
sep()   { echo -e "${CYAN}──────────────────────────────────────────────────────────${NC}"; }

# ── Parse args ───────────────────────────────────────────────────────────────
MODE="docker"
for arg in "$@"; do
  [[ "$arg" == "--local" ]] && MODE="local"
  [[ "$arg" == "--docker" ]] && MODE="docker"
done

sep
echo -e "  ${BOLD}AnomalyNetFlow — Setup${NC}  (mode: ${MODE})"
sep

# =============================================================================
# DOCKER PATH (default)
# =============================================================================
if [[ "$MODE" == "docker" ]]; then

  step "Checking Docker"
  if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker Desktop from https://docs.docker.com/get-docker/
     then re-run:  ./setup.sh
     Or run without Docker:  ./setup.sh --local"
  fi
  ok "Docker $(docker --version | head -1)"

  if ! docker info &>/dev/null; then
    fail "Docker daemon is not running. Start Docker Desktop and try again."
  fi
  ok "Docker daemon running"

  # ── .env ─────────────────────────────────────────────────────────────────
  step "Environment configuration"
  if [ -f ".env" ]; then
    ok ".env already exists — skipping"
  else
    cp .env.demo .env
    ok ".env created from .env.demo"
  fi

  # ── Dataset check ─────────────────────────────────────────────────────────
  step "Checking CIC-IDS2017 dataset"
  CSV_COUNT=$(find -L data/CIC-IDS2017 -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
  if [ "$CSV_COUNT" -gt 0 ]; then
    ok "Found ${CSV_COUNT} CSV file(s) in data/CIC-IDS2017/"
  else
    GDRIVE_FILE_ID=$(grep -E "^CICIDS2017_GDRIVE_FILE_ID=" .env 2>/dev/null | cut -d= -f2 | tr -d '[:space:]' || true)
    if [ -z "$GDRIVE_FILE_ID" ] && [ -f ".env.demo" ]; then
      GDRIVE_FILE_ID=$(grep -E "^CICIDS2017_GDRIVE_FILE_ID=" .env.demo | cut -d= -f2 | tr -d '[:space:]')
    fi
    if [ -n "$GDRIVE_FILE_ID" ]; then
      info "Downloading CIC-IDS2017 dataset (Google Drive)..."
      mkdir -p data/CIC-IDS2017
      if docker run --rm -v "$SCRIPT_DIR/data:/data" \
          python:3.11-slim \
          bash -c "pip install gdown -q && gdown --id '$GDRIVE_FILE_ID' -O /data/CIC-IDS2017.zip && cd /data && unzip -q CIC-IDS2017.zip -d CIC-IDS2017/ && rm CIC-IDS2017.zip" 2>/dev/null; then
        CSV_COUNT=$(find -L data/CIC-IDS2017 -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
        [ "$CSV_COUNT" -gt 0 ] && ok "Downloaded ${CSV_COUNT} CSV file(s)" || \
          warn "Download succeeded but CSV files not found — check data/CIC-IDS2017/"
      else
        warn "Auto-download failed. Place CIC-IDS2017 CSV files in data/CIC-IDS2017/ manually."
        info "Source: https://www.unb.ca/cic/datasets/ids-2017.html"
      fi
    else
      warn "No CIC-IDS2017 CSV files. Place them in data/CIC-IDS2017/ before training."
      info "Source: https://www.unb.ca/cic/datasets/ids-2017.html"
    fi
  fi

  # ── Build + start ─────────────────────────────────────────────────────────
  step "Building and starting containers (docker compose up --build)"
  info "This may take a few minutes on first run (downloading base images + building layers)."
  info "Subsequent starts skip the build and take ~10 seconds."
  echo ""
  docker compose up --build -d

  # Wait for health
  info "Waiting for API health check..."
  local_up=0
  for _ in $(seq 1 12); do
    sleep 5
    if curl -sf "http://localhost:8000/health" >/dev/null 2>&1; then
      local_up=1; break
    fi
  done

  sep
  if [ "$local_up" -eq 1 ]; then
    ok "AnomalyNetFlow is running!"
  else
    warn "API did not become healthy within 60 s — check logs: docker compose logs api"
  fi
  sep
  echo ""
  echo -e "  ${BOLD}Access the application:${NC}"
  echo ""
  echo -e "    ${CYAN}http://localhost:8080${NC}        — Web UI"
  echo -e "    ${CYAN}http://localhost:8000/docs${NC}   — REST API (Swagger)"
  echo ""
  echo -e "  ${BOLD}First time?${NC}  Open the Training tab → click ${CYAN}Train on CIC-IDS2017${NC}"
  echo ""
  echo -e "  ${BOLD}Useful commands:${NC}"
  echo -e "    docker compose logs -f api     — stream API logs"
  echo -e "    docker compose stop            — stop all containers"
  echo -e "    docker compose down            — stop + remove containers"
  echo ""
  sep
  exit 0
fi

# =============================================================================
# LOCAL PATH (--local flag)
# =============================================================================

CONDA_ENV="dl_env"
PYTHON_VERSION="3.11"
NODE_MIN=20

step "Checking prerequisites"

if ! command -v conda &>/dev/null; then
  fail "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
fi
ok "conda $(conda --version)"

if ! command -v node &>/dev/null; then
  fail "Node.js not found. Install from https://nodejs.org (version ${NODE_MIN}+)"
fi
NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt "$NODE_MIN" ]; then
  fail "Node.js ${NODE_MIN}+ required, found $(node --version)"
fi
ok "Node.js $(node --version)"

# ── Conda env ─────────────────────────────────────────────────────────────────
step "Setting up Python environment (conda: ${CONDA_ENV})"
if conda env list | grep -qE "^${CONDA_ENV}\s"; then
  ok "Conda environment '${CONDA_ENV}' already exists"
else
  info "Creating conda env '${CONDA_ENV}' with Python ${PYTHON_VERSION} ..."
  conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
  ok "Environment created"
fi

# ── Python deps ───────────────────────────────────────────────────────────────
step "Installing Python dependencies"
info "Running: pip install -r requirements.txt"
conda run -n "$CONDA_ENV" pip install -r requirements.txt --quiet
ok "Python dependencies installed"

# ── UI deps ───────────────────────────────────────────────────────────────────
step "Installing UI dependencies (npm)"
cd "$SCRIPT_DIR/ui"
npm install --silent
ok "UI dependencies installed"
cd "$SCRIPT_DIR"

# ── .env ──────────────────────────────────────────────────────────────────────
step "Environment configuration"
if [ -f ".env" ]; then
  ok ".env already exists — skipping"
else
  cp .env.demo .env
  ok ".env created from .env.demo"
  info "Edit .env to change defaults (thresholds, paths, etc.)"
fi

# ── Runtime dirs ──────────────────────────────────────────────────────────────
step "Creating runtime directories"
mkdir -p models logs
touch models/.gitkeep logs/.gitkeep
ok "models/  logs/  ready"

# ── Dataset ───────────────────────────────────────────────────────────────────
step "Checking CIC-IDS2017 dataset"
CSV_COUNT=$(find -L data/CIC-IDS2017 -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -gt 0 ]; then
  ok "Found ${CSV_COUNT} CSV file(s) in data/CIC-IDS2017/"
else
  warn "No CSV files found in data/CIC-IDS2017/"

  GDRIVE_FILE_ID=""
  if [ -f ".env" ]; then
    GDRIVE_FILE_ID=$(grep -E "^CICIDS2017_GDRIVE_FILE_ID=" .env | cut -d= -f2 | tr -d '[:space:]')
  fi
  if [ -z "$GDRIVE_FILE_ID" ] && [ -f ".env.demo" ]; then
    GDRIVE_FILE_ID=$(grep -E "^CICIDS2017_GDRIVE_FILE_ID=" .env.demo | cut -d= -f2 | tr -d '[:space:]')
  fi

  _dataset_manual_instructions() {
    warn "Place the CIC-IDS2017 CSV files manually in data/CIC-IDS2017/ before training."
    info "Source: https://www.unb.ca/cic/datasets/ids-2017.html"
    info "(Look for 'MachineLearningCVE' or 'GeneratedLabelledFlows' in the archive)"
  }

  if [ -n "$GDRIVE_FILE_ID" ]; then
    info "Downloading CIC-IDS2017.zip from Google Drive ..."
    mkdir -p data/CIC-IDS2017
    if conda run -n "$CONDA_ENV" gdown --id "$GDRIVE_FILE_ID" -O data/CIC-IDS2017.zip 2>/dev/null; then
      info "Extracting ..."
      unzip -q data/CIC-IDS2017.zip -d data/CIC-IDS2017/ 2>/dev/null || true
      rm -f data/CIC-IDS2017.zip
      CSV_COUNT=$(find -L data/CIC-IDS2017 -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
      if [ "$CSV_COUNT" -gt 0 ]; then
        ok "Downloaded and extracted ${CSV_COUNT} CSV file(s) into data/CIC-IDS2017/"
      else
        warn "Download succeeded but no CSV files found."
        _dataset_manual_instructions
      fi
    else
      warn "Google Drive download failed."
      _dataset_manual_instructions
    fi
  else
    info "To enable auto-download, set CICIDS2017_GDRIVE_FILE_ID in .env.demo"
    _dataset_manual_instructions
  fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
sep
echo -e "  ${BOLD}${GREEN}Setup complete.${NC}"
sep
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo ""
echo -e "    1.  ${CYAN}bash ops.sh start${NC}   — start the API  (port 8000)"
echo -e "    2.  ${CYAN}bash ops.sh ui${NC}      — start the UI   (port 5173, separate terminal)"
echo -e "    3.  Open ${CYAN}http://localhost:5173${NC} and click  Train on CIC-IDS2017"
echo ""
echo -e "    API docs:  ${CYAN}http://localhost:8000/docs${NC}"
echo ""
sep
