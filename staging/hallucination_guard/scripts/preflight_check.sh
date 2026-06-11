#!/bin/bash
# =============================================================================
# Hallucination Guard — Pre-flight Safety Check
# =============================================================================
# Run BEFORE any modification to 8001 PROD.
# Verifies: VCSAI safety, service health, env state, disk space, backup space.
# Exit code 0 = safe to proceed. Non-zero = STOP, investigate.
# =============================================================================

set -u  # don't set -e; we want to continue past individual failures and report

APP_DIR="/home/ubuntu/chatbot/aniruddha/vcsai/unified-rag-agent-v31"
SERVICE="rag-agent.service"
TOKEN_LOG_DIR="${APP_DIR}/_token_logs"
BACKUP_DIR="/home/ubuntu/backups/hallucination_guard"
MIN_FREE_GB=5

PASS=0
FAIL=0
WARN=0

check_pass() { echo "  [PASS] $1"; PASS=$((PASS+1)); }
check_fail() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
check_warn() { echo "  [WARN] $1"; WARN=$((WARN+1)); }

echo "=============================================================================="
echo "Hallucination Guard — Pre-flight Check"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host:      $(hostname)"
echo "App dir:   ${APP_DIR}"
echo "=============================================================================="

# -----------------------------------------------------------------------------
# 1. App directory exists
# -----------------------------------------------------------------------------
echo ""
echo "[1] App directory check"
if [ -d "${APP_DIR}" ]; then
  check_pass "${APP_DIR} exists"
else
  check_fail "${APP_DIR} missing — wrong host?"
  exit 10
fi

# -----------------------------------------------------------------------------
# 2. Service health
# -----------------------------------------------------------------------------
echo ""
echo "[2] Service status (${SERVICE})"
if systemctl is-active --quiet "${SERVICE}" 2>/dev/null; then
  check_pass "${SERVICE} is active"
  UPTIME=$(systemctl show "${SERVICE}" --property=ActiveEnterTimestamp --value 2>/dev/null)
  echo "       (since ${UPTIME})"
else
  check_warn "${SERVICE} NOT active — proceed only if intentional"
fi

# -----------------------------------------------------------------------------
# 3. VCSAI safety — check for active in-process jobs
# -----------------------------------------------------------------------------
echo ""
echo "[3] VCSAI in-process jobs (must be zero before restart)"
if [ -d "${TOKEN_LOG_DIR}" ]; then
  # heuristic: jsonl files modified in last 60s indicate active streaming jobs
  ACTIVE=$(find "${TOKEN_LOG_DIR}" -name "*.jsonl" -mmin -1 2>/dev/null | wc -l)
  if [ "${ACTIVE}" -eq 0 ]; then
    check_pass "No active jobs in last 60s"
  else
    check_warn "${ACTIVE} jsonl(s) modified in last 60s — wait and re-run"
    find "${TOKEN_LOG_DIR}" -name "*.jsonl" -mmin -1 2>/dev/null | head -5
  fi
else
  check_warn "Token log dir not found (${TOKEN_LOG_DIR}) — VCSAI may not be deployed"
fi

# -----------------------------------------------------------------------------
# 4. Disk space (need room for backups + eval logs)
# -----------------------------------------------------------------------------
echo ""
echo "[4] Disk space (need >= ${MIN_FREE_GB} GB free)"
FREE_GB=$(df -BG "${APP_DIR}" | awk 'NR==2 {gsub("G",""); print $4}')
if [ "${FREE_GB}" -ge "${MIN_FREE_GB}" ]; then
  check_pass "Free space: ${FREE_GB}G"
else
  check_fail "Only ${FREE_GB}G free — need ${MIN_FREE_GB}G+. Free up space first."
fi

# -----------------------------------------------------------------------------
# 5. Critical files exist
# -----------------------------------------------------------------------------
echo ""
echo "[5] Critical files present"
for f in ".env" "registry.py" "agentic/tools/registry.py"; do
  if [ -f "${APP_DIR}/${f}" ] || [ -f "${APP_DIR}/${f%.*}*" ]; then
    check_pass "${f} (or variant) exists"
  else
    check_warn "${f} not found (may be normal if structure differs)"
  fi
done
# At least one synthesizer file should exist
SYNTH_FILES=$(find "${APP_DIR}" -maxdepth 4 -name "synth*.py" -o -name "*synthesizer*.py" 2>/dev/null | head -5)
if [ -n "${SYNTH_FILES}" ]; then
  check_pass "synthesizer files found:"
  echo "${SYNTH_FILES}" | sed 's/^/         /'
else
  check_warn "No synthesizer-named files; inspect manually"
fi

# -----------------------------------------------------------------------------
# 6. Current env flag state
# -----------------------------------------------------------------------------
echo ""
echo "[6] Current .env state (relevant flags)"
if [ -f "${APP_DIR}/.env" ]; then
  for flag in ENABLE_SPECIFICATIONS HALLUCINATION_GUARD_ENABLED HG_PILLAR_1 HG_PILLAR_4 HG_PILLAR_6 HG_PILLAR_7 HG_PILLAR_11; do
    VAL=$(grep -E "^${flag}=" "${APP_DIR}/.env" 2>/dev/null | head -1)
    if [ -n "${VAL}" ]; then
      echo "       ${VAL}"
    else
      echo "       ${flag}  [not set — defaults to OFF]"
    fi
  done
  check_pass ".env readable"
else
  check_fail ".env not found at ${APP_DIR}/.env"
fi

# -----------------------------------------------------------------------------
# 7. Backup directory ready
# -----------------------------------------------------------------------------
echo ""
echo "[7] Backup directory (${BACKUP_DIR})"
if [ ! -d "${BACKUP_DIR}" ]; then
  mkdir -p "${BACKUP_DIR}" 2>/dev/null && check_pass "Created ${BACKUP_DIR}" || check_fail "Cannot create ${BACKUP_DIR}"
else
  check_pass "${BACKUP_DIR} exists"
fi
BAK_FREE_GB=$(df -BG "${BACKUP_DIR}" | awk 'NR==2 {gsub("G",""); print $4}')
echo "       Backup volume free: ${BAK_FREE_GB}G"

# -----------------------------------------------------------------------------
# 8. Python + pip availability (for eval suite + verifier deps)
# -----------------------------------------------------------------------------
echo ""
echo "[8] Python / dependency check"
if command -v python3 >/dev/null 2>&1; then
  PYVER=$(python3 --version 2>&1)
  check_pass "${PYVER}"
else
  check_fail "python3 not on PATH"
fi
if python3 -c "import requests, json" 2>/dev/null; then
  check_pass "requests, json importable"
else
  check_warn "requests not installed — needed for eval suite (pip install requests)"
fi

# -----------------------------------------------------------------------------
# 9. ai_assistant.pem NOT in current dir (HARD RULE check)
# -----------------------------------------------------------------------------
echo ""
echo "[9] PROD-key isolation check"
if [ -f "${APP_DIR}/ai_assistant.pem" ]; then
  check_fail "ai_assistant.pem present in app dir — HARD RULE VIOLATION; remove before proceeding"
else
  check_pass "ai_assistant.pem not in app dir"
fi

# -----------------------------------------------------------------------------
# 10. /query endpoint smoke test (no flag changes)
# -----------------------------------------------------------------------------
echo ""
echo "[10] Endpoint smoke (GET / or /health)"
if curl -sf --max-time 5 http://127.0.0.1:8001/health >/dev/null 2>&1 || \
   curl -sf --max-time 5 http://127.0.0.1:8001/ >/dev/null 2>&1; then
  check_pass "8001 reachable on localhost"
else
  check_warn "8001 not responding on localhost (may be normal if behind different binding)"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================================================="
echo "Pre-flight summary"
echo "=============================================================================="
echo "  PASS:  ${PASS}"
echo "  WARN:  ${WARN}"
echo "  FAIL:  ${FAIL}"
echo ""

if [ "${FAIL}" -gt 0 ]; then
  echo "RESULT: STOP — ${FAIL} failure(s). Investigate before proceeding."
  exit 1
elif [ "${WARN}" -gt 0 ]; then
  echo "RESULT: PROCEED WITH CAUTION — ${WARN} warning(s). Review and decide."
  exit 2
else
  echo "RESULT: SAFE TO PROCEED. All pre-flight checks passed."
  exit 0
fi
