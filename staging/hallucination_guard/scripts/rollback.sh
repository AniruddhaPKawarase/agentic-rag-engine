#!/bin/bash
# =============================================================================
# Hallucination Guard — Rollback to Snapshot
# =============================================================================
# One-command restore from a baseline snapshot.
# Usage: ./rollback.sh <snapshot_tag>           # restore specific snapshot
#        ./rollback.sh                           # restore LATEST
# Recovers: .env + critical .py files + tarball if needed.
# Restarts: rag-agent.service after restore.
# =============================================================================

set -euo pipefail

APP_DIR="/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31"
SERVICE="rag-agent.service"
BACKUP_DIR="/home/ubuntu/backups/hallucination_guard"

# -----------------------------------------------------------------------------
# Resolve snapshot tag
# -----------------------------------------------------------------------------
if [ $# -ge 1 ]; then
  TAG="$1"
else
  if [ -L "${BACKUP_DIR}/LATEST" ]; then
    TAG=$(readlink "${BACKUP_DIR}/LATEST")
    echo "No tag given — using LATEST: ${TAG}"
  else
    echo "ERROR: no tag given and no LATEST symlink. Usage:"
    echo "  $0 <snapshot_tag>"
    echo ""
    echo "Available snapshots:"
    ls -1d "${BACKUP_DIR}"/hg_v2_baseline_* 2>/dev/null | xargs -n1 basename || echo "  (none)"
    exit 2
  fi
fi

SNAPSHOT_DIR="${BACKUP_DIR}/${TAG}"

if [ ! -d "${SNAPSHOT_DIR}" ]; then
  echo "ERROR: snapshot directory not found: ${SNAPSHOT_DIR}"
  exit 3
fi

echo "=============================================================================="
echo "Hallucination Guard — Rollback"
echo "Snapshot:   ${TAG}"
echo "Source:     ${SNAPSHOT_DIR}"
echo "Target:     ${APP_DIR}"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# 1. Verify checksums
# -----------------------------------------------------------------------------
echo ""
echo "[1] Verifying snapshot integrity..."
if [ -f "${SNAPSHOT_DIR}/checksums.sha256" ]; then
  ( cd "${SNAPSHOT_DIR}" && sha256sum -c checksums.sha256 --quiet ) && echo "       OK" || { echo "       FAIL — checksums mismatch. ABORT."; exit 4; }
else
  echo "       WARN: no checksums file; proceeding anyway"
fi

# -----------------------------------------------------------------------------
# 2. Show what we'd restore (dry-run display)
# -----------------------------------------------------------------------------
echo ""
echo "[2] Files to restore:"
FILES_TO_RESTORE=$(find "${SNAPSHOT_DIR}" -type f ! -name "SNAPSHOT_META.txt" ! -name "checksums.sha256" ! -name "full_app.tar.gz" | sort)
echo "${FILES_TO_RESTORE}" | sed "s|${SNAPSHOT_DIR}/||" | sed 's/^/       /'

echo ""
echo "[3] Confirm restore? Type 'YES-ROLLBACK' to proceed:"
read -r CONFIRMATION
if [ "${CONFIRMATION}" != "YES-ROLLBACK" ]; then
  echo "Aborted by user."
  exit 5
fi

# -----------------------------------------------------------------------------
# 4. Stop service (be polite — graceful)
# -----------------------------------------------------------------------------
echo ""
echo "[4] Stopping ${SERVICE}..."
sudo systemctl stop "${SERVICE}" || true
sleep 2

# -----------------------------------------------------------------------------
# 5. Restore files
# -----------------------------------------------------------------------------
echo ""
echo "[5] Restoring files..."
RESTORED=0
while IFS= read -r SRC; do
  REL_PATH="${SRC#${SNAPSHOT_DIR}/}"
  DST="${APP_DIR}/${REL_PATH}"
  DST_DIR="$(dirname "${DST}")"
  mkdir -p "${DST_DIR}"
  cp -p "${SRC}" "${DST}"
  echo "       restored ${REL_PATH}"
  RESTORED=$((RESTORED+1))
done <<< "${FILES_TO_RESTORE}"

echo ""
echo "       Total files restored: ${RESTORED}"

# -----------------------------------------------------------------------------
# 6. Restart service
# -----------------------------------------------------------------------------
echo ""
echo "[6] Restarting ${SERVICE}..."
sudo systemctl start "${SERVICE}"
sleep 5

if systemctl is-active --quiet "${SERVICE}"; then
  echo "       ${SERVICE} active"
else
  echo "       WARN: ${SERVICE} not active — investigate logs:"
  echo "         journalctl -u ${SERVICE} -n 50"
  exit 6
fi

# -----------------------------------------------------------------------------
# 7. Smoke test
# -----------------------------------------------------------------------------
echo ""
echo "[7] Smoke test..."
sleep 2
if curl -sf --max-time 10 http://127.0.0.1:8001/health >/dev/null 2>&1 || \
   curl -sf --max-time 10 http://127.0.0.1:8001/ >/dev/null 2>&1; then
  echo "       8001 responding"
else
  echo "       WARN: 8001 not responding on localhost — check logs"
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "Rollback complete."
echo "Restored from: ${TAG}"
echo "Service:       ${SERVICE} active"
echo ""
echo "Next steps:"
echo "  1. Verify /query endpoint with a known-good test query"
echo "  2. Review journalctl -u ${SERVICE} -n 100 for any startup errors"
echo "  3. Run a quick eval to confirm pre-patch behavior restored"
echo "=============================================================================="
