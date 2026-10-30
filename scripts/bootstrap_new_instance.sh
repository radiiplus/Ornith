#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendor_rev=cba0325974f809ff8f2c07c39d2f2ec2a4173293
model_rev=1fc60d6805eaa18bbed4088391db9336eb9581c4
model_hash=88ade210cf31da75951b1e86e00ad6d656232c705721d427884af3434c097dd2
model_path="$root/models/ornith-1.5-9b-iq2_m.gguf"

for command in git cmake c++ make curl sha256sum bwrap python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing $command. On Amazon Linux 2023, install prerequisites with:" >&2
    echo "  sudo dnf install git cmake gcc-c++ make curl bubblewrap python3" >&2
    exit 1
  fi
done

if [[ ! -d "$root/vendor/llama.cpp/.git" ]]; then
  mkdir -p "$root/vendor"
  git clone https://github.com/shuxiaoqiong/llama.cpp.git "$root/vendor/llama.cpp"
fi
actual_rev="$(git -C "$root/vendor/llama.cpp" rev-parse HEAD)"
if [[ "$actual_rev" != "$vendor_rev" ]]; then
  if [[ -n "$(git -C "$root/vendor/llama.cpp" status --porcelain)" ]]; then
    echo 'The llama.cpp checkout has local changes at another revision; inspect it first.' >&2
    exit 1
  fi
  git -C "$root/vendor/llama.cpp" checkout --detach "$vendor_rev"
fi
if git -C "$root/vendor/llama.cpp" apply --check "$root/experiments/patches/llama.cpp-local.patch"; then
  git -C "$root/vendor/llama.cpp" apply "$root/experiments/patches/llama.cpp-local.patch"
elif ! git -C "$root/vendor/llama.cpp" apply --reverse --check "$root/experiments/patches/llama.cpp-local.patch"; then
  echo 'The local llama.cpp patch does not match this checkout; inspect it first.' >&2
  exit 1
fi

cmake -S "$root/vendor/llama.cpp" -B "$root/vendor/llama.cpp/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build "$root/vendor/llama.cpp/build" -j 4 \
  --target llama-server llama-cli llama-bench

mkdir -p "$root/models"
if ! printf '%s  %s\n' "$model_hash" "$model_path" | sha256sum --check --status 2>/dev/null; then
  curl --fail --location --retry 3 --continue-at - \
    --output "$model_path" \
    "https://huggingface.co/bartowski/Ornith-1.5-9B-GGUF/resolve/$model_rev/Ornith-1.5-9B-IQ2_M.gguf"
fi
printf '%s  %s\n' "$model_hash" "$model_path" | sha256sum --check --status
PYTHONPATH="$root/src" python3 -m unittest discover -s "$root/tests"
echo 'Bootstrap complete. Read docs/new-instance-handoff.md before changing CPU limits.'
