#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/src"
exec systemd-run --user --scope --quiet \
  -p CPUQuota=50% -p MemoryHigh=4G -p MemoryMax=6G \
  -p MemorySwapMax=19G -p IOWeight=10 \
  /usr/bin/timeout --signal=TERM --kill-after=10s 1200s \
  python3 "$root/scripts/evaluate_candidate_roles.py" "$@"
