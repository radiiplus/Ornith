# Stage 3 inference engine baseline

Measured 2026-10-29 UTC. Raw runs: [normal](../artifacts/stage3-baseline.json) and [2 GiB pressure](../artifacts/stage3-pressure-2gib.json). Xing4.0 weights were not downloaded.

## Engine decision

The pinned [Xing4.0 llama.cpp port](https://github.com/shuxiaoqiong/llama.cpp/tree/cba0325974f809ff8f2c07c39d2f2ec2a4173293) is the baseline CPU engine. It contains model-specific mHC, MLA, and MoE code and a CPU backend; [upstream PR #29012](https://github.com/ggml-org/llama.cpp/pull/29012) was still open at inspection. This is architecture-specific support, not an inference from generic GGUF support. The publisher's [vLLM](https://github.com/XingChen-AGI/xing4-vllm-plugin) and [SGLang](https://github.com/XingChen-AGI/xing4-sglang-plugin) plugins support Xing4.0 but target GPU execution and much larger memory footprints; they are not viable baselines on this GPU-less 7.60 GiB RAM host. Plain upstream llama.cpp is not selected because Xing4.0 support is unmerged. The port remains a maintenance risk until upstreamed; pin the revision and keep an adapter boundary.

The engine was built **unchanged**, CPU-only, from commit `cba0325974f809ff8f2c07c39d2f2ec2a4173293` with CMake Release, `GGML_NATIVE=ON`, `LLAMA_CURL=OFF`, and tests disabled. Targets built: `llama-cli`, `llama-bench`, `llama-server`. Compiler GCC 11.5.0; CMake 3.22.2; Git 2.50.1; reported engine `0.7.0-dev` build 1. The binary and source are under `vendor/llama.cpp`; no weights are copied into that tree. The server binds only to `127.0.0.1`, disables its web UI, uses mmap by default, and has one slot. No model-specific runtime modifications were made.

## Test model and method

The only model file is the publisher's [Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) at revision `f86cb2c1fa58255f8052cc32aeede1b7482d4361`, 1,117,320,768 bytes, SHA-256 `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`. It is an infrastructure baseline, not a proxy for Xing4.0 quality or speed. Each run used 2 CPU threads, 2048 context tokens, one parallel slot, an identical 26-token coding prompt, deterministic temperature 0, and two 128-token streamed generations. The second request reused a cached prompt. Time to first token is measured from HTTP request to first nonempty text chunk. Decode rate is `(completion_tokens - 1) / (response_end - first_token)` using server usage counts. Model-load time is from the server's `load_model: loading model` to `model loaded` log events. Startup time is process launch to a healthy HTTP response.

| Metric | Normal | 2 GiB pressure |
| --- | ---: | ---: |
| Startup | 1.547 s | 1.466 s |
| Model load (log events) | 1.465 s | 1.402 s |
| First-token latency, request 1 / 2 | 0.362 / 0.075 s | 0.364 / 0.120 s |
| Decode rate, request 1 / 2 | 8.54 / 8.58 tokens/s | 8.54 / 8.51 tokens/s |
| Peak engine RSS | 2,185,904,128 B (2.04 GiB) | 2,186,096,640 B (2.04 GiB) |
| Peak engine virtual memory | 2.58 GiB | 2.58 GiB |
| Peak system swap / engine swap | 0 / 0 B | 0 / 0 B |
| Engine major-fault delta | 0 | 0 |
| Engine physical disk-read delta | 0 B | 0 B |
| Minimum system MemAvailable | 5.83 GiB | 3.83 GiB |
| Memory PSI `some` / `full` avg10 | 0.00 / 0.00 | 0.00 / 0.00 |
| Two 128-token generations | completed | completed |

The 2 GiB pressure reservation was touched in the benchmark process and held during engine startup and generation; it was released at exit. It did not force swap or measurable PSI. Model pages were likely warm in the page cache after download and prior runs, explaining the zero process disk-read and major-fault deltas. Those counters are process-scoped, sampled from `/proc/<pid>/io` and `/proc/<pid>/stat`; they do not include file reads by other processes or guarantee a cold-start measurement. RSS is the sampled `VmHWM` high-water mark. The generated response reached the token limit; this benchmark did not score answer correctness. No OOM stress or cache-dropping system change was performed.

## Reproduction and interface

The standard-library-only [harness](../scripts/benchmark_inference.py) verifies GGUF magic, model SHA-256, and a conservative available-RAM budget before launch. It reports missing files, unsupported formats, engine startup failures, and insufficient memory explicitly. `InferenceEngine` defines `start`, `wait_ready`, `generate`, and `stop`; `LlamaCppAdapter` is the current implementation. The HTTP streaming contract and JSON result remain stable when adding a new engine adapter. Context length, generation limit, CPU threads, repetitions, and bounded memory pressure are command-line settings. Full commands and software versions are recoverable from the harness arguments and JSON artifacts.

Example normal run from the project root:

```bash
python3 scripts/benchmark_inference.py \
  --server vendor/llama.cpp/build/bin/llama-server \
  --model models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --model-revision f86cb2c1fa58255f8052cc32aeede1b7482d4361 \
  --model-quantization Q4_K_M \
  --model-sha256 cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046 \
  --engine-revision cba0325974f809ff8f2c07c39d2f2ec2a4173293 \
  --context 2048 --max-tokens 128 --threads 2 --repetitions 2 \
  --output artifacts/stage3-baseline.json
```

Add `--pressure-mib 2048 --output artifacts/stage3-pressure-2gib.json` for the pressure case. Run them sequentially. Do not compare these results with other models, context lengths, quantizations, CPU settings, or cache states as if they were equivalent.

**Next gate:** The small-model baseline shows no measured engine limitation requiring a code change. Xing4.0 feasibility remains unproven because even its smallest publisher GGUF exceeds RAM. Before acquiring one Xing4.0 file, review available disk and the 4.5 GiB model-resident planning budget, then define acceptable latency and a bounded stop condition for a short-context run. Retain the 30 GiB root free-space reserve. Do not use these Qwen numbers to predict Xing4.0 throughput.
