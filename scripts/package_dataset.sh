#!/usr/bin/env bash
# =============================================================================
# Package CIC-IDS2017 CSVs into a single zip for Google Drive sharing.
#
# Usage:
#   bash scripts/package_dataset.sh
#
# Output:
#   CIC-IDS2017.zip  (created in the project root)
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/CIC-IDS2017"
OUT_ZIP="$PROJECT_DIR/CIC-IDS2017.zip"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✔${NC}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }
fail() { echo -e "  ${RED}✘${NC}  $*"; exit 1; }
info() { echo -e "  ${CYAN}→${NC}  $*"; }

echo -e "\n${BOLD}Package CIC-IDS2017 dataset${NC}\n"

# Check source directory
if [ ! -d "$DATA_DIR" ]; then
  fail "Dataset directory not found: $DATA_DIR"
fi

CSV_COUNT=$(find -L "$DATA_DIR" -maxdepth 1 -name "*.csv" | wc -l)
if [ "$CSV_COUNT" -eq 0 ]; then
  fail "No CSV files found in $DATA_DIR"
fi
ok "Found ${CSV_COUNT} CSV file(s) in data/CIC-IDS2017/"

# Remove existing zip
if [ -f "$OUT_ZIP" ]; then
  warn "Removing existing $OUT_ZIP"
  rm "$OUT_ZIP"
fi

# Create zip — store CSVs flat inside a CIC-IDS2017/ folder in the archive
info "Creating $OUT_ZIP ..."
cd "$PROJECT_DIR"
zip -j "$OUT_ZIP" "$DATA_DIR"/*.csv
ok "Created: $OUT_ZIP"

# Print size
SIZE=$(du -sh "$OUT_ZIP" | cut -f1)
info "Archive size: ${BOLD}${SIZE}${NC}"

echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo -e "  1. Upload ${CYAN}CIC-IDS2017.zip${NC} to Google Drive"
echo -e "  2. Right-click → Share → ${BOLD}Anyone with the link${NC} → Copy link"
echo -e "  3. Extract the file ID from the share URL:"
echo -e "       https://drive.google.com/file/d/${CYAN}<FILE_ID>${NC}/view"
echo -e "  4. Add to ${CYAN}.env${NC}:"
echo -e "       CICIDS2017_GDRIVE_FILE_ID=<FILE_ID>"
echo -e "  5. Commit ${CYAN}.env.demo${NC} with the file ID so the examiner"
echo -e "     gets it automatically on  ${CYAN}bash setup.sh${NC}"
echo ""
