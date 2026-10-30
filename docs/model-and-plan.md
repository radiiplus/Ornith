# Model compatibility and staged plan

Research checked 2026-10-29 UTC. Repository listings and PR status can change; pin revisions before installation. No model files were downloaded.

## Model facts

The [publisher's model card](https://github.com/XingChen-AGI/Xing4.0-29B-A4B) describes 29B total and about 4B active parameters, 40 layers, 3584 hidden width, MLA attention, 64 routed experts with 4 selected per token, one shared expert, and mHC/MTP components. The [Hugging Face BF16 repository](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/tree/a32a2e99b5877c6b7ad16e9954ec08488bcc5fa2) is labeled Apache-2.0 and totals about 62.4 GB. The [FP8 repository](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8/tree/main) is also labeled Apache-2.0 and totals about 33.2 GB. Repository totals include ancillary files and are not peak resident-memory measurements. The effective active parameter count does not reduce the storage needed for all routed experts.

The [publisher's GGUF files](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-GGUF/tree/main) are Apache-2.0 labeled and include IQ2_M-slim 11.8 GB, IQ3_XXS 14 GB, IQ3_M 15.4 GB, IQ4_XS 18.3 GB, IQ4_NL 20.1 GB, and larger variants. These are decimal GB as displayed by Hugging Face. The repository currently has multiple variants, so a single-file selection is essential. Lower precision may degrade code quality; no task-specific quality comparison has been established for this instance.

The [publisher's deployment guide](https://github.com/XingChen-AGI/Xing4.0-29B-A4B) lists Transformers custom code and vLLM/SGLang plugins. Its sample vLLM/SGLang deployments use GPUs. The [Xing4.0 llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/29012) remains open and states CPU backend, conversion, and quantization support. Thus stock upstream support cannot be assumed. A pinned Xing4.0-capable `llama.cpp` branch is the most plausible CPU evaluation path, pending source review and a build. The PR author's published speed figures are not measurements on this 2-vCPU host. Hugging Face's generic "use with" snippets are not proof that every stock engine handles this architecture.

**Feasibility:** The smallest listed publisher GGUF exceeds physical RAM before runtime, KV cache, OS, and agent processes. A 20 GiB swap target could permit allocation but cannot ensure usable speed, especially during random expert access. BF16 and FP8 are unsuitable as initial CPU-only candidates. Begin with a short context (for example 2K-4K), one request, and memory mapping; derive actual peak RSS, page faults, disk I/O, tokens/s, and response quality before attempting longer context or service operation. A 256K context claim is a model capability, not a feasible operating setting here.

## Proposed project structure

```text
xing-specialist/
  docs/                  decisions, runbooks, compatibility and benchmarks
  artifacts/             small snapshots and benchmark summaries
  configs/               pinned model and runtime profiles
  src/xing_specialist/   orchestration, tools, evaluation, memory interfaces
  tests/                 focused unit and integration tests
  scripts/               repeatable setup and benchmark commands
```

Only `docs/` and `artifacts/` have been created for this assessment. Future weight storage should be a single external, durable path on the root volume, not in the repository or `/tmp`; record checksums and pin model/engine revisions. Reuse one Hugging Face cache or direct GGUF path without duplicate copies.

## Stages and gates

1. **Review environment:** Decide whether Amazon Linux is acceptable or provision Ubuntu; establish a durable, writable project/model location. Confirm root-volume budget and swap policy. Gate: OS, storage, and system-change decision recorded.
2. **Prepare host:** Install minimal compiler/build tools and version control with pinned versions. If approved, create about 20 GiB swap on persistent storage and set swappiness to 100; verify reboot persistence and disk headroom. Gate: compiler smoke test, swap and mount checks.
3. **Validate engine without weights:** Pin and review the Xing4.0-capable `llama.cpp` revision; build CPU backend and run its unit/operator tests. Gate: model architecture recognized and test results recorded.
4. **Acquire one quantized checkpoint:** Choose the smallest quality-acceptable publisher GGUF after checking exact bytes, license, checksum/source revision, disk headroom, and expected process memory. Download recoverably to the persistent volume; verify integrity. Gate: no duplicate weights and successful metadata inspection.
5. **Measure inference:** Short prompt and 2K-4K context first; record peak RSS/swap, major faults, I/O, prompt and decode throughput, and correctness on compiler/debugging tasks. Gate: explicit minimum usable latency and quality thresholds chosen from results. Stop or change hardware/model if untenable.
6. **Build specialist workflow:** Implement constrained repo inspection, patching, test execution, evidence capture, and recovery checkpoints around the measured engine. Add a small held-out suite for compilers, language implementation, systems work, performance, root-cause analysis, and test generation. Gate: reproducible runs and no unverified success claims.
7. **Improve cautiously:** Store evaluations and approved examples; compare prompt/tool changes against held-out tasks before promotion. Fine-tuning only after memory, compute, data rights, and measurable benefit are established. Gate: regression results and rollback path.

## Uncertainties and blockers

- **Host mismatch:** User specified AWS Ubuntu; actual host is Amazon Linux. Package commands and reproducibility differ.
- **Persistence/access:** Current writable root is `/tmp` tmpfs. A durable writable location on `/` is needed before installation or model acquisition.
- **Memory/performance:** No swap, no GPU, and 2 vCPUs. CPU decode speed and page-fault behavior for 11.8-20.1 GB GGUF remain unmeasured.
- **Engine maturity:** Xing4.0 `llama.cpp` support is an unmerged PR. CPU build and correctness on this host remain unverified.
- **Quantization tradeoff:** The smallest GGUF has no local compiler-task evaluation; low-bit degradation may be material.
- **Runtime envelope:** Exact KV cache, graph/workspace, and peak RSS for the selected branch and context remain unknown.
- **Security and trust:** Transformers requires publisher custom code; review any pinned remote code before execution. Agent command execution needs explicit sandbox and repository scope in implementation.
- **Self-improvement meaning:** Define approval, held-out evaluation, and rollback criteria before any automatic prompt or weight changes.

This stage ends here pending review. No tests were run because no code or engine was installed.
