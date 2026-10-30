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
`experiments/patches/llama.cpp-local.patch`, builds the CPU server, downloads
`Ornith-1.5-9B-IQ2_M.gguf` from pinned Hugging Face revision
`1fc60d6805eaa18bbed4088391db9336eb9581c4`, verifies SHA-256
`88ade210cf31da75951b1e86e00ad6d656232c705721d427884af3434c097dd2`,
and runs the Python suite. A second 4.72 GB IQ3_M model existed locally but was not
needed for the current bounded agent runs and is not restored by this script.

## Current evidence

- Ornith IQ2_M is the target. Under a 50% single-CPU quota, 4 GiB soft/6 GiB hard
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
