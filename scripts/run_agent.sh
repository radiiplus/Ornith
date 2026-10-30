#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$root/src"
exec python3 -m xing_specialist.cli "$@"
