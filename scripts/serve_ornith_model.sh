#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec systemd-run --user --scope --quiet \
  -p CPUQuota=50% \
  -p MemoryHigh=4G \
  -p MemoryMax=6G \
  -p MemorySwapMax=19G \
  -p IOWeight=10 \
  /usr/bin/timeout --signal=TERM --kill-after=10s 1800s \
  "$root/vendor/llama.cpp/build/bin/llama-server" \
  --model "$root/models/ornith-1.5-9b-iq2_m.gguf" \
  --ctx-size 2048 --parallel 2 --threads 1 --threads-batch 1 \
  --batch-size 128 --ubatch-size 64 --load-mode mmap --fit on \
  --no-warmup --numa isolate --reasoning off \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --host 127.0.0.1 --port 8080 --no-webui
