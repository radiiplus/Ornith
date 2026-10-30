#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$root/.venv/bin/python" ]]; then
  echo 'Create .venv and install configs/lora-pilot-requirements.txt first' >&2
  exit 2
fi
exec systemd-run --user --scope --quiet \
  -p CPUQuota=100% -p MemoryHigh=4G -p MemoryMax=6G \
  -p MemorySwapMax=19G -p IOWeight=10 \
  /usr/bin/timeout --signal=TERM --kill-after=10s 600s \
  "$root/.venv/bin/python" "$root/scripts/pilot_lora_reasoning.py" "$@"
