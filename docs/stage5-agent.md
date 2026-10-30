# Stage 5 agent and Xing4.0 acquisition

Implemented and tested 2026-10-29 UTC. The host remains Amazon Linux 2023, with 7.60 GiB RAM, an existing ~20 GiB swap file, and swappiness 200. The earlier user choice was to leave that value unchanged. No system memory setting changed in this stage.

## Xing4.0 file

The single publisher IQ2_M-slim GGUF is at `models/xing4_0-29b-iq2_M-slim.gguf`: 11,800,813,760 bytes, SHA-256 `eab889116ae30ab32c3d2948f42409c4e2a4f238b89904b6976448cd067be312`, matching the [publisher listing](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-GGUF/tree/195d1bd04080d73bdead6f40d8d7d1a8f43d5050). About 64 GiB of root disk remained after acquisition. This is one weight copy, outside Git and outside `/tmp`. Download and checksum verification do **not** establish CPU-load feasibility, speed, or IQ2 coding quality. The [Stage 2 memory estimate](memory_estimates.md) still applies: near-full residency plus host reserves exceeds physical RAM. The temporary Qwen Q4_K_M file remains the working agent backend. Both exact file identities are in [model-manifest.json](../configs/model-manifest.json).

## Components

| Component | Implementation | Contract |
| --- | --- | --- |
| Model backend | `src/xing_specialist/backend.py` | `ModelBackend.generate(messages, limits)`; local OpenAI-compatible HTTP adapter, configurable context/output/timeout, explicit HTTP/JSON errors |
| Planner | `planner.py` | Five verifiable stages with prerequisites; refuses the same failed change without new evidence and caps attempts |
| Tool execution | `tools.py` | Argument-list command policy, Bubblewrap copy of repository, no network, clean environment, CPU/address-space/file/time limits, captured exit/stdout/stderr/duration |
| Memory | `memory.py` | SQLite tasks, plan steps, hypotheses, actions, observations, test results, lessons, evidence links; WAL and secret redaction |
| Retrieval | `retrieval.py` | SQLite FTS5/BM25 over lessons accepted only with passing test evidence; CPU-only, no embedding/FAISS dependency |
| Verification | `verification.py` | Runs approved tests, records observed pass/fail and baseline regressions; model text cannot mark a test passed |
| Repair loop | `agent.py`, `workspace.py` | Inspect approved source, form hypothesis, stage exact edit or bounded diff, test, revise within budget, store evidence, optionally promote and retest |

The default does **not** modify the source repository. `--apply` is an explicit promotion option after staged tests pass. Automatic promotion requires a failing baseline test that the staged change fixes, and refuses approved paths under `tests/` or with a `test_` basename. A passing baseline cannot establish that the objective was met, so such proposals remain `needs_review` and produce no verified lesson. A failure after promotion is marked `needs_review`; the agent does not silently undo source changes. An exact-text edit must match once in an approved file; diffs are size-limited and path-checked. Model-generated shell commands are not executed. Test commands are supplied by the caller and still pass the policy. Outputs are redacted and capped at 64 KiB per field, so very large raw logs are not retained. The sandbox copies at most 50 MiB / 3000 files and excludes model weights, vendor code, credentials by known names, and artifacts. Unsupported repositories fail closed. Sandbox memory is limited by process virtual address space (default 2048 MiB); this is a conservative process limit, not a whole-cgroup cap. Bubblewrap user, mount, PID and network namespaces isolate test code. Credentials are not inherited into the sandbox.

The agent stores retrieval and verified lessons, not updated model parameters. A lesson records the objective, approved test, files, and passing evidence; it does not claim broad correctness outside those tests. Model choice is independent of planner, tools, memory, retrieval, and verification. To substitute a future Xing4.0 server, provide its local API URL and model ID after a separate load/performance gate.

## Verification

Unit and integration tests cover backend limits/errors, planner dependencies and retry prevention, SQLite persistence and evidence gating, FTS5 retrieval, command rejection, sandbox isolation and timeout, patch-path validation, exact edits, regression recording, and staged/promotion behavior. The real temporary Qwen backend answered a direct request. On `examples/repair-demo`, its first diff response had incorrect hunk counts and was rejected; a repeated failed diff was stopped without source changes. After adding a constrained exact-text edit option, the real model proposed `return a - b` to `return a + b`; the agent applied it in staging, passed the approved unittest, recorded a linked lesson, and returned `needs_review` without changing the example source. The SQLite audit for that smoke run is under `artifacts/agent-smoke/`.

No Xing4.0 inference run, long-task evaluation, adversarial prompt-injection test, or reboot test was performed. The next model gate is a short, memory-guarded Xing4.0 load with explicit stop thresholds and measured page faults, disk reads, swap, and tokens/s. If it is unusable, retain the temporary backend and evaluate a different quantization or hardware rather than changing the agent.
