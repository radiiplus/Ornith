# New-instance handoff

The public GitHub repository contains source, docs, tests, curated benchmark JSON,
and a patch for two local changes to llama.cpp. It deliberately excludes GGUF weights,
the 338 MB vendor checkout, Rust build output, SQLite audit databases, and credentials.
There is no durable agent database in `~/.local/share/xing-specialist` on the old host;
the task results to continue from are the committed artifacts and this handoff. A new
Codex session must be started in the clone and instructed to read this file.

## Restore

On an Amazon Linux 2023 instance with enough disk for the model and build:

```bash
git clone https://github.com/radiiplus/Ornith.git
cd Ornith
sudo dnf install git cmake gcc-c++ make curl bubblewrap python3
bash scripts/bootstrap_new_instance.sh
```

Bootstrap checks out `shuxiaoqiong/llama.cpp` at
`cba0325974f809ff8f2c07c39d2f2ec2a4173293`, applies
`experiments/patches/llama.cpp-local.patch`, builds the CPU server, and downloads
the pinned VibeThinker and DeepSeek Q4_K_M GGUFs for the current
[two-model experiment](two-model-router.md). Pass `--include-ornith` to also
download `Ornith-1.5-9B-IQ2_M.gguf` from pinned Hugging Face revision
`1fc60d6805eaa18bbed4088391db9336eb9581c4`, verifies SHA-256
`88ade210cf31da75951b1e86e00ad6d656232c705721d427884af3434c097dd2`,
and runs the Python suite. A second 4.72 GB IQ3_M model existed locally but was not
needed for the current bounded agent runs and is not restored by this script.

## Current evidence

- Ornith IQ2_M is the historical baseline. Under a 50% single-CPU quota, 4 GiB soft/6 GiB hard
  memory bounds, and one generation thread, a memory-guided slug repair passed staged
  tests at 0.517 generated tokens/s. Identical verified replay used no model call and
  still reran tests. See `docs/ornith-adaptive-stream-eval.md` and its artifacts.
- The reasoning stream exposed model-provided text, not hidden internal state. On a
  simple workflow task its first answer was wrong; checker feedback corrected it on
  the second attempt. See `artifacts/ornith-stream-probe-feedback-rescored.json`.
- More demanding arithmetic and counting tasks still failed in the bounded stress
  results. No weight update or general intelligence gain has been demonstrated.
- The Rust context probe is experimental. The deployed agent uses SQLite retrieval.
  The agent does not yet resume an interrupted model generation from a checkpoint.

## Next work on 16 vCPUs

Measure the same Ornith prompts at 4, 6, 8, and 12 generation threads before choosing
the smallest setting near peak throughput. `CPUQuota=800%` is an initial eight-vCPU
ceiling; `CPUQuota=80%` would still allow only 0.8 of one vCPU. The existing launchers
and `require_envelope()` in `scripts/evaluate_ornith_prompt.py` enforce 50%, so their
checks must be updated together for a larger host. Keep a hard memory limit, enough
RAM for active model pages, a swap ceiling, a deadline, and host-pressure monitoring.
Do not run multiple full model copies while benchmarking. Start with
`scripts/run_ornith_adaptive_eval.sh` and `scripts/run_ornith_stream_probe.sh`, then
repeat reasoning checks on unseen tasks before changing training data or weights.

The public repository is a code and experiment handoff. It cannot transfer this chat
session or resume a partly generated response automatically. Read `AGENTS.md`, run the
tests, restore the model, and continue from the measured limitations above.

## Current two-model experiment

The active work now uses VibeThinker-1.5B Q4_K_M for objective restatement and
planning, and DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M for bounded actions and
repairs. `scripts/bootstrap_new_instance.sh` restores these two models by
default; Ornith is optional with `--include-ornith`. The source-backed
`corpus/hunt/` files cover three existing verified fixture patterns. The
router protocol, limits, and benchmark artifacts are in
`docs/two-model-router.md`.

The corrected combined nine-case suite passed 4/9 in 1,077.17 seconds at a
100% one-core quota, 5.12 effective generated tokens/s, and 2,084.7 MiB peak
model RSS. Its seven reasoning cases passed 3/7, below DeepSeek alone's 4/7
at 50% CPU; tenant isolation still failed staged repair. See
`artifacts/dual-model-combined-final.json`. A separate capped VibeThinker
LoRA pilot updated 544,768 parameters in 112.36 seconds but did not improve
held-out loss or the 1/7 standalone exact-checker score. It remains disabled;
see `docs/lora-pilot.md`.

On this two-vCPU host, the seven exact reasoning checks passed 1/7 for
VibeThinker and 4/7 for DeepSeek at a 50% CPU quota. The actual repair demo
passed for VibeThinker after two proposals and for DeepSeek after its edit
path was constrained; both standalone tenant-isolation runs failed. A
128-token speed probe measured roughly 3 tokens/s per model at 50% CPU and
6.6-6.8 tokens/s at a 100% one-core quota with one thread. The combined
runner uses the latter setting with the same 4/6 GiB RAM and 19 GiB swap
bounds. The larger host should remeasure thread count and quota rather than
reuse the two-vCPU optimum. SQLite task memory is local state and is not
included in Git; committed artifacts are the portable evaluation record.

The next independent candidate pass is recorded in
`docs/small-model-candidate-evaluation.md`. Qwen3-1.7B Q4_K_M and Granite
4.0 H-1B each passed 4/7 checked reasoning cases; Gemma 3 1B passed 1/7.
All three passed the small repair demo but failed tenant isolation. The
existing router remains on VibeThinker and DeepSeek until objective fidelity
and a larger unseen set justify a change. Youtu-LLM-2B has no quality score:
the current llama.cpp CPU backend cannot execute its MLA graph. Candidate
GGUFs are not in Git; the report records pinned sources and SHA-256 hashes.
