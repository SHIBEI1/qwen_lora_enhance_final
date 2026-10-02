#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
qde_enable_log "verify_release"
"$QDE_PYTHON" "$QDE_PROJECT_ROOT/scripts/verify_release.py" --project-root "$QDE_PROJECT_ROOT" "$@"
