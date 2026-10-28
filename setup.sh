#!/usr/bin/env bash
# =============================================================================
# AnomalyNetFlow — First-time Setup
#
# Run once before starting the application:
#   bash setup.sh
#
# Prerequisites (the only things you need before running this):
#   1. Miniconda or Anaconda  https://docs.conda.io/en/latest/miniconda.html
#   2. Node.js >= 20          https://nodejs.org
#   3. CIC-IDS2017 CSV files  placed in  data/CIC-IDS2017/
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="dl_env"
PYTHON_VERSION="3.11"
NODE_MIN=20

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

step()  { echo -e "\n${BOLD}${CYAN}▶ $*${NC}"; }
ok()    { echo -e "  ${GREEN}✔${NC}  $*"; }
warn()  { echo -e "  ${YELLOW}⚠${NC}  $*"; }
fail()  { echo -e "  ${RED}✘${NC}  $*"; exit 1; }
info()  { echo -e "     $*"; }

# ── Separator ─────────────────────────────────────────────────────────────────
sep() { echo -e "${CYAN}──────────────────────────────────────────────────────────${NC}"; }

sep
echo -e "  ${BOLD}AnomalyNetFlow — Setup${NC}"
sep

# ── 1. Check Conda ─────────────────────────────────────────────────────────
step "Checking prerequisites"

if ! command -v conda &>/dev/null; then
  fail "conda not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
fi
ok "conda $(conda --version)"

# ── 2. Check Node.js ───────────────────────────────────────────────────────
if ! command -v node &>/dev/null; then
  fail "Node.js not found. Install from https://nodejs.org (version ${NODE_MIN}+)"
fi
NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt "$NODE_MIN" ]; then
  fail "Node.js ${NODE_MIN}+ required, found v$(node --version | sed 's/v//')"
fi
ok "Node.js $(node --version)"

# ── 3. Create / update Conda environment ──────────────────────────────────
step "Setting up Python environment (conda: ${CONDA_ENV})"

if conda env list | grep -qE "^${CONDA_ENV}\s"; then
  ok "Conda environment '${CONDA_ENV}' already exists"
else
  info "Creating conda env '${CONDA_ENV}' with Python ${PYTHON_VERSION} ..."
  conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
  ok "Environment created"
fi

# ── 4. Install Python dependencies ─────────────────────────────────────────
step "Installing Python dependencies"
info "Running: pip install -r requirements.txt"
conda run -n "$CONDA_ENV" pip install -r requirements.txt --quiet
ok "Python dependencies installed"

# ── 5. Install UI dependencies ─────────────────────────────────────────────
step "Installing UI dependencies (npm)"
cd "$SCRIPT_DIR/ui"
npm install --silent
ok "UI dependencies installed"
cd "$SCRIPT_DIR"

# ── 6. Create .env if missing ──────────────────────────────────────────────
step "Environment configuration"
if [ -f ".env" ]; then
  ok ".env already exists — skipping"
else
  cp .env.example .env
  ok ".env created from .env.example"
  info "Edit .env to change defaults (thresholds, paths, etc.)"
fi

# ── 7. Create required runtime directories ────────────────────────────────
step "Creating runtime directories"
mkdir -p models logs
touch models/.gitkeep logs/.gitkeep
ok "models/  logs/  ready"

# ── 8. Check for CIC-IDS2017 dataset ──────────────────────────────────────
step "Checking CIC-IDS2017 dataset"
CSV_COUNT=$(find -L data/CIC-IDS2017 -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -gt 0 ]; then
  ok "Found ${CSV_COUNT} CSV file(s) in data/CIC-IDS2017/"
else
  warn "No CSV files found in data/CIC-IDS2017/"
  info "Download the CIC-IDS2017 dataset and place the CSV files there before training."
  info "Source: https://www.unb.ca/cic/datasets/ids-2017.html"
  info "(Look for 'MachineLearningCVE' or 'GeneratedLabelledFlows' in the archive)"
fi

# ── Done ──────────────────────────────────────────────────────────────────
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
