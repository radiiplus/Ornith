#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
exec "$root/vendor/llama.cpp/build/bin/llama-server" \
  --model "$root/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" \
  --ctx-size 2048 --threads 2 --threads-batch 2 --parallel 1 \
  --host 127.0.0.1 --port 8080 --no-webui --load-mode auto
