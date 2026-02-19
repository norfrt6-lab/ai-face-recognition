#!/usr/bin/env bash
# ============================================================
# AI Face Recognition & Face Swap
# scripts/healthcheck.sh — Local development health check
# ============================================================
# Polls the API health endpoint and reports component status.
#
# Usage:
#   bash scripts/healthcheck.sh              # default localhost:8000
#   bash scripts/healthcheck.sh http://api:8000   # custom URL
# ============================================================

set -euo pipefail

API_URL="${1:-http://localhost:8000}"
HEALTH_ENDPOINT="${API_URL}/api/v1/health"

echo "Checking API health at: ${HEALTH_ENDPOINT}"
echo "-------------------------------------------"

RESPONSE=$(curl -sf --max-time 5 "${HEALTH_ENDPOINT}" 2>&1) || {
    echo "FAIL: API is not reachable at ${API_URL}"
    echo "      Make sure the server is running:"
    echo "        uvicorn api.main:app --host 0.0.0.0 --port 8000"
    exit 1
}

# Pretty-print if python is available, otherwise raw output
if command -v python3 &>/dev/null; then
    echo "${RESPONSE}" | python3 -m json.tool
elif command -v python &>/dev/null; then
    echo "${RESPONSE}" | python -m json.tool
else
    echo "${RESPONSE}"
fi

# Check status field
STATUS=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")

echo "-------------------------------------------"
if [ "${STATUS}" = "ok" ]; then
    echo "RESULT: API is healthy"
    exit 0
elif [ "${STATUS}" = "degraded" ]; then
    echo "RESULT: API is degraded (some components not loaded)"
    exit 0
else
    echo "RESULT: API is unhealthy (status=${STATUS})"
    exit 1
fi
