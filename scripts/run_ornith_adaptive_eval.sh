#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec systemd-run --user --scope --quiet \
  -p CPUQuota=50% \
  -p MemoryHigh=4G \
  -p MemoryMax=6G \
  -p MemorySwapMax=19G \
  -p IOWeight=10 \
  /usr/bin/timeout --signal=TERM --kill-after=10s 1200s \
  python3 "$script_dir/evaluate_ornith_adaptive.py" "$@"
