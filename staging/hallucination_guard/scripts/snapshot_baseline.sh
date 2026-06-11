#!/bin/bash
# =============================================================================
# Hallucination Guard — Baseline Snapshot
# =============================================================================
# Creates a tagged backup of the current 8001 PROD state BEFORE any modification.
# Run AFTER preflight_check.sh exits 0 or 2 (PASS or PROCEED-WITH-CAUTION).
# Exit code 0 = snapshot created. Non-zero = STOP, fix, retry.
# =============================================================================

set -euo pipefail

APP_DIR="/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31"
BACKUP_DIR="/home/ubuntu/backups/hallucination_guard"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="hg_v2_baseline_${TIMESTAMP}"
SNAPSHOT_DIR="${BACKUP_DIR}/${TAG}"

echo "=============================================================================="
echo "Hallucination Guard — Baseline Snapshot"
echo "Tag:        ${TAG}"
echo "Snapshot:   ${SNAPSHOT_DIR}"
echo "=============================================================================="

mkdir -p "${SNAPSHOT_DIR}"

# -----------------------------------------------------------------------------
# 1. Manifest of files we're snapshotting
# -----------------------------------------------------------------------------
echo ""
echo "[1] Building file manifest..."

# Files most likely to be touched by hallucination guard:
declare -a TARGETS=(
  ".env"
  "registry.py"
  "agentic/tools/registry.py"
  "agentic/agentic_core.py"
  "agentic/runner.py"
  "main.py"
  "app.py"
  "routes.py"
  "handlers.py"
)

# Glob: any synthesizer/synth file
SYNTH_FILES=$(find "${APP_DIR}" -maxdepth 5 -type f \( -name "synth*.py" -o -name "*synthesizer*.py" \) 2>/dev/null || true)

# -----------------------------------------------------------------------------
# 2. Copy targets (file-by-file with structure preserved)
# -----------------------------------------------------------------------------
echo ""
echo "[2] Copying targeted files..."
COPIED=0
for f in "${TARGETS[@]}"; do
  SRC="${APP_DIR}/${f}"
  if [ -f "${SRC}" ]; then
    REL_DIR="$(dirname "${f}")"
    DST_DIR="${SNAPSHOT_DIR}/${REL_DIR}"
    mkdir -p "${DST_DIR}"
    cp -p "${SRC}" "${DST_DIR}/"
    echo "       ${f}"
    COPIED=$((COPIED+1))
  fi
done

if [ -n "${SYNTH_FILES}" ]; then
  while IFS= read -r SRC; do
    REL_PATH="${SRC#${APP_DIR}/}"
    REL_DIR="$(dirname "${REL_PATH}")"
    DST_DIR="${SNAPSHOT_DIR}/${REL_DIR}"
    mkdir -p "${DST_DIR}"
    cp -p "${SRC}" "${DST_DIR}/"
    echo "       ${REL_PATH}"
    COPIED=$((COPIED+1))
  done <<< "${SYNTH_FILES}"
fi

echo ""
echo "       Total files copied: ${COPIED}"

# -----------------------------------------------------------------------------
# 3. Whole-app tarball (full safety net; ~10-50 MB typically)
# -----------------------------------------------------------------------------
echo ""
echo "[3] Creating full app tarball..."
tar --exclude='__pycache__' \
    --exclude='_token_logs' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='node_modules' \
    -czf "${SNAPSHOT_DIR}/full_app.tar.gz" \
    -C "$(dirname "${APP_DIR}")" \
    "$(basename "${APP_DIR}")"

TAR_SIZE=$(du -h "${SNAPSHOT_DIR}/full_app.tar.gz" | cut -f1)
echo "       full_app.tar.gz: ${TAR_SIZE}"

# -----------------------------------------------------------------------------
# 4. Record metadata
# -----------------------------------------------------------------------------
echo ""
echo "[4] Recording metadata..."
cat > "${SNAPSHOT_DIR}/SNAPSHOT_META.txt" <<META
Snapshot Tag:        ${TAG}
Timestamp (UTC):     $(date -u +%Y-%m-%dT%H:%M:%SZ)
Host:                $(hostname)
App Dir:             ${APP_DIR}
Snapshot Dir:        ${SNAPSHOT_DIR}
Created By:          ${USER}
Reason:              Pre-Hallucination Guard v2 deployment baseline
Files Copied:        ${COPIED}
Service Status:      $(systemctl is-active rag-agent.service 2>/dev/null || echo "unknown")

Current .env hallucination-guard flags:
$(grep -E "^(HALLUCINATION_GUARD_ENABLED|HG_PILLAR_)" "${APP_DIR}/.env" 2>/dev/null || echo "  [none — all default OFF]")

Current .env critical flags:
$(grep -E "^(ENABLE_SPECIFICATIONS|MASTER_SPECS_OFF|PERSONA_V32_ENABLED)" "${APP_DIR}/.env" 2>/dev/null || echo "  [none found]")

To roll back:
    ./rollback.sh ${TAG}

To verify integrity:
    sha256sum -c ${SNAPSHOT_DIR}/checksums.sha256
META

# Checksums for tamper detection
( cd "${SNAPSHOT_DIR}" && find . -type f ! -name "checksums.sha256" -exec sha256sum {} + > checksums.sha256 )

echo "       Metadata: ${SNAPSHOT_DIR}/SNAPSHOT_META.txt"
echo "       Checksums: ${SNAPSHOT_DIR}/checksums.sha256"

# -----------------------------------------------------------------------------
# 5. Update LATEST symlink
# -----------------------------------------------------------------------------
echo ""
echo "[5] Updating LATEST symlink..."
ln -sfn "${TAG}" "${BACKUP_DIR}/LATEST"
echo "       ${BACKUP_DIR}/LATEST -> ${TAG}"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "Snapshot complete. Tag: ${TAG}"
echo "Total snapshot size: $(du -sh "${SNAPSHOT_DIR}" | cut -f1)"
echo ""
echo "ROLLBACK COMMAND (one-liner):"
echo "  bash $(dirname "$0")/rollback.sh ${TAG}"
echo "=============================================================================="
