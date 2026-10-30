# Small-data LoRA feasibility pilot

This is a controlled weight-update experiment on VibeThinker-1.5B, separate
from the default two-model router. It leaves the original BF16 checkpoint and
Q4_K_M GGUF untouched. The adapter is not enabled because it did not improve
the local checks.

## Inputs and envelope

The base checkpoint is `WeiboAI/VibeThinker-1.5B` at revision
`9614bcb0264f03f18f2fa2406c4a9754635f51de`. Its
`model.safetensors` SHA-256 is
`4fe09d459343b3c5f0f147c8ab7dcf9c86677658cbd01f44d7bcd60f6d08a730`.
Training uses the official [GSM8K](https://github.com/openai/grade-school-math)
train split (SHA-256
`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`),
and the official test split solely for held-out loss (SHA-256
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`).
Only rows fitting 128 tokens were sampled; this is a tiny, length-biased
sample, not a representative training run. The model uses BF16 CPU weights,
rank-4 LoRA on `q_proj` and `v_proj`, gradient checkpointing, one PyTorch
thread, and a 100% one-core quota. The launcher enforces 4 GiB soft/6 GiB
hard memory, 19 GiB swap, and a ten-minute deadline. The upstream tokenizer
lacks a chat template, so the pilot supplies the system/user/assistant Qwen
role delimiters present in the local GGUF's template. Only the adapter is
saved under ignored `models/`.

## Measurements

| Pilot | Steps | Learning rate | Wall time | Peak RSS | Four-example held-out loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Feasibility | 1 | 1e-4 | 6.74 s | 3,668.6 MiB | not measured |
| Short sample | 4 | 1e-4 | 22.64 s | 3,703.4 MiB | 6.1952 to 6.2202 |
| Longer sample | 32 | 1e-5 | 112.36 s | 3,897.3 MiB | 6.1952 to 6.2019 |

The 32-step adapter has 544,768 trainable parameters and a 2.2 MiB GGUF
export. Loaded over the unchanged Q4_K_M base, it passed the same 1/7 exact
reasoning cases as the unmodified model, with the same sole pass
(`converse-fallacy`). Sum of request time was 134.58 s versus 133.61 s for
the unmodified baseline at a 50% CPU quota. The four held-out loss values
are too small a sample to predict broader quality, but both pilot variants
worsened that measure, and the exact checks show no gain. Neither adapter is
part of the default router. See `artifacts/vibethinker-gsm8k-lora-*.json` and
`artifacts/vibethinker-standalone-lora-32step.json`.

## Reproduction

Install Python 3.11 and its pip module, create `.venv`, then install
`torch==2.5.1+cpu` from the PyTorch CPU wheel index and the pinned packages in
`configs/lora-pilot-requirements.txt`. Download the pinned BF16 snapshot into
`models/vibethinker-1.5b-bf16`, and the official GSM8K train and test JSONL
files into a local cache. The script verifies both dataset hashes before use.
For the longer sample:

```bash
bash scripts/run_lora_pilot.sh \
  --model-dir models/vibethinker-1.5b-bf16 \
  --data /home/ec2-user/.cache/ornith/gsm8k/train.jsonl \
  --eval-data /home/ec2-user/.cache/ornith/gsm8k/test.jsonl \
  --adapter-dir models/vibethinker-gsm8k-lora-32step \
  --output artifacts/vibethinker-gsm8k-lora-32step.json \
  --max-steps 32 --sequence-length 128 --learning-rate 0.00001
```

The adapter can be converted with vendored `convert_lora_to_gguf.py` and
checked through `run_ornith_stress.sh --lora`. Keep training data, BF16 weights,
the adapter, and GGUF export out of Git. The artifact records measurements,
not a claim of improved reasoning.
