#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"

# Keep profiled runs lightweight by default. Override LOG_CACHE_STATE=1 only
# when cache debugging is explicitly needed.
export LOG_CACHE_STATE="${LOG_CACHE_STATE:-0}"

bash "$SCRIPT_DIR/run_profiled_inference.sh" bash "$SCRIPT_DIR/run_smoke_causal_camera.sh"
