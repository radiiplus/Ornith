#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
role="${1:-}"
quota="${2:-}"
threads="${3:-}"
spec_type="${4:-none}"
flash_attn="${5:-auto}"
label="${6:-}"
if [[ "$quota" != 50 && "$quota" != 100 ]] || [[ "$threads" != 1 && "$threads" != 2 ]]; then
  echo 'Usage: bash scripts/run_small_model_speed.sh reasoning|execution 50|100 1|2 [none|ngram-simple] [auto|on|off] [label]' >&2
  exit 2
fi
if [[ "$spec_type" != none && "$spec_type" != ngram-simple ]]; then
  echo 'Speculative type must be none or ngram-simple' >&2
  exit 2
fi
if [[ "$flash_attn" != auto && "$flash_attn" != on && "$flash_attn" != off ]]; then
  echo 'Flash attention must be auto, on, or off' >&2
  exit 2
fi
if [[ -n "$label" && ! "$label" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'Label must contain only letters, digits, underscore, or hyphen' >&2
  exit 2
fi
suffix=""
if [[ "$spec_type" != none ]]; then
  suffix="-${spec_type}"
fi
if [[ "$flash_attn" != auto ]]; then
  suffix="${suffix}-flash-${flash_attn}"
fi
if [[ -n "$label" ]]; then
  suffix="${suffix}-${label}"
fi
case "$role" in
  reasoning)
    model="$root/models/vibethinker-1.5b-q4_k_m.gguf"
    name=VibeThinker-1.5B
    revision=864584a2e09fc02d59d694bfb8a67ed1060593eb
    hash=3df5dae7a65dfd426c8dc58d97ed1247af4b669bc2674c5f9b2b34103b01e164
    ;;
  execution)
    model="$root/models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf"
    name=DeepSeek-R1-Distill-Qwen-1.5B
    revision=9cc28b17e86fa2415fcb070f8ee5ec27c965aa61
    hash=1741e5b2d062b07acf048bf0d2c514dadf2a48f94e2b4aa0cfe069af3838ee2f
    ;;
  *)
    echo 'Role must be reasoning or execution' >&2
    exit 2
    ;;
esac
exec systemd-run --user --scope --quiet \
  -p "CPUQuota=${quota}%" -p MemoryHigh=4G -p MemoryMax=6G \
  -p MemorySwapMax=19G -p IOWeight=10 \
  /usr/bin/timeout --signal=TERM --kill-after=10s 300s \
  python3 "$root/scripts/benchmark_inference.py" \
    --server "$root/vendor/llama.cpp/build/bin/llama-server" \
    --model "$model" --model-name "$name" --model-revision "$revision" \
    --model-quantization Q4_K_M --model-sha256 "$hash" \
    --engine-revision cba0325974f809ff8f2c07c39d2f2ec2a4173293 \
    --context 2048 --max-tokens 128 --threads "$threads" --repetitions 1 \
    --load-mode mmap --no-warmup --batch-size 128 --ubatch-size 64 \
    --max-anonymous-mib 2304 --min-available-mib 1536 \
    --max-swap-used-mib 19456 --max-cgroup-full-pressure 5 \
    --spec-type "$spec_type" \
    --flash-attn "$flash_attn" \
    --output "$root/artifacts/speed-${role}-${quota}pct-${threads}t${suffix}.json"
