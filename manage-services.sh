#!/usr/bin/env bash
# Compatibility shim.  Execution authority lives in profiles.json/brokerctl.py.

set -euo pipefail

ROOT="${LLAMA_BROKER_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

case "${1:-status}" in
    models) set -- list "${@:2}" ;;
    start) set -- switch "${@:2}" ;;
    stop-all) set -- stop "${@:2}" ;;
esac

exec python3 "$ROOT/brokerctl.py" "$@"
