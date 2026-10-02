#!/usr/bin/env bash
# Run the readiness check with the project's fixed preflight log.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

qde_enable_log "preflight"
"$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/preflight.py" \
  --project-root "$QDE_PROJECT_ROOT" \
  --require-model \
  "$@"
