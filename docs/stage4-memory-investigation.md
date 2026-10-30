# Stage 4 memory-aware inference investigation

Inspected 2026-10-29 UTC at pinned Xing4.0 llama.cpp port commit `cba0325974f809ff8f2c07c39d2f2ec2a4173293`. Measured artifacts: [mmap auto](../artifacts/stage4-mmap-auto.json), [no mmap](../artifacts/stage4-no-mmap.json), [correctness auto](../artifacts/stage4-correctness-auto.json), [correctness none](../artifacts/stage4-correctness-none.json), and [guard failure](../artifacts/stage4-guard-test.json). The only loaded model was Qwen2.5-Coder-1.5B-Instruct Q4_K_M; Xing4.0 weights remain absent.

## Existing engine controls

The CPU engine's `--load-mode auto` selects mmap when the backend supports it ([loader](../vendor/llama.cpp/src/llama-model.cpp)); `--load-mode mmap` and `none` make this explicit. On Linux, the [mmap implementation](../vendor/llama.cpp/src/llama-mmap.cpp) uses read-only `MAP_SHARED`, advises sequential access, may request `MAP_POPULATE`, and uses `POSIX_MADV_WILLNEED` for non-lazy ranges. This can warm much of a weight file at startup; mmap is not a promise of low RSS or zero disk I/O. The model loader maps tensor regions rather than requiring a second full in-memory weight copy. `mlock` would prevent reclaim and is unsuitable for Xing4.0 on 7.60 GiB RAM. `none` allocates/copied weight buffers and raises anonymous-memory exposure.

Clean mapped model pages are file-backed and reclaimable: the kernel can discard them and later reread the GGUF from SSD. The same virtual mapping can have different resident page counts as the cache changes. Anonymous KV, graph, and tensor buffers are not reclaimed that way; they may consume swap or trigger OOM. Model-file pages also compete with the OS page cache. The engine already exposes `--ctx-size`, `--cache-type-k/v`, `--batch-size`, `--ubatch-size`, `--parallel`, and thread counts ([arguments](../vendor/llama.cpp/common/arg.cpp)). The baseline fixes one slot and a 2048-token context, bounding concurrency and cache capacity; generation is limited by the harness. Smaller batch and microbatch are available if measured prefill peaks require them. KV quantization would need a correctness and memory comparison on Xing4.0 before selection.

## Matched small-model comparison

Both runs used the same pinned engine, Qwen file/hash, 2 threads, 2048 context, one slot, two 128-token generations, warm page cache, and a 4096 MiB anonymous-memory guard. These are single runs, not a statistical performance study.

| Measure | `auto` mmap | `none` |
| --- | ---: | ---: |
| Engine startup | 1.55 s | 2.28 s |
| Model-load log interval | 1.45 s | 2.22 s |
| Mapped model-file virtual bytes at ready | 919,937,024 B | 0 B |
| Resident bytes in those model-file mappings | 919,937,024 B | 0 B |
| Peak engine RSS | 2,185,867,264 B | 1,276,088,320 B |
| Peak anonymous memory | 1,121,304,576 B | 1,260,068,864 B |
| Peak file-backed PSS | 1,128,834,048 B | 12,868,608 B |
| Decode rate, request 1 / 2 | 8.12 / 8.07 tokens/s | 8.27 / 8.16 tokens/s |
| Peak swap / major-fault delta / process disk reads | 0 / 0 / 0 B | 0 / 0 / 0 B |

The lower RSS with `none` reflects the absence of resident *file-backed* pages, not an overall memory win: anonymous memory grew by about 139 MB and loading slowed by about 0.76 s. The small decode-rate difference is not a reproducible benefit. Separate deterministic 128-token checks with both modes produced the same SHA-256, `0a1a1fbeb5b5605beddfe0030feea6cace0db27c8cd36ed51fad2322e83ac682`, and both ended at the configured token limit. This establishes equality only for that prompt and length, not broad model parity. **Keep mmap auto; no engine code change is justified.**

All runs were cache-warm after downloading and earlier benchmarks. Zero process `read_bytes` and major-fault deltas do not prove that the SSD would be idle with an 11.8 GB Xing4.0 file. The model mapping measurement reads `/proc/<pid>/maps` and `smaps`; anonymous and file-backed proportional memory come from `smaps_rollup`. RSS includes reclaimable file pages, while `VmSize` includes unresident address space. The 0.2-second sampling interval can miss brief allocation peaks; use the engine's `VmHWM` as the RSS high-water mark.

## Prefetching and MoE

The engine already provides OS readahead hints for mapped weights and a `--lazy-mode` facility for *certain* tensors, such as per-layer embeddings. Its existing `llama_prefetch_rows` path is not a general Xing4.0 expert cache. In [Xing4.0's model graph](../vendor/llama.cpp/src/models/xing4_0.cpp), routed experts are packed as expert tensors; the router chooses four of 64 at runtime, and the shared expert executes for every MoE layer. Future layers are sequential, but the routed expert identities are input-dependent. Prefetching every expert would overread; predicting selected experts early would require validated overlap after router results and could change memory pressure. Per-expert eviction would need new tensor-storage semantics and must preserve all experts for future tokens. There is no observed Xing4.0 I/O trace to justify either change. No asynchronous-read or expert-cache prototype was added.

## Pressure guard and recovery

The [harness](../scripts/benchmark_inference.py) now samples anonymous memory, system MemAvailable, total swap use, RSS, major faults, and process disk reads. Defaults are 2048 MiB maximum engine anonymous memory, 1024 MiB minimum system MemAvailable, and 1024 MiB maximum system swap use. Each is configurable. A breached guard terminates the local server, writes a failed JSON artifact, and exits with a concise error. The deliberate 128 MiB anonymous-limit test stopped the engine; `stage4-guard-test.json` records the failure and no server process remained. This is a **best-effort sampled stop condition**, not a kernel-enforced hard cap: transient peaks can occur between samples and OS-wide swap may come from other processes. Keep kernel reclaim and swap management; no custom swap manager was added.

## Decision and next evidence

Maintain `auto` mmap, one parallel slot, short context, and the tested guard defaults. No engine optimization was retained because the small-model baseline has no measured I/O or swap bottleneck. The next meaningful Xing4.0 investigation needs one approved quantized file, a short bounded run, model-specific page-fault and disk-read traces, and a deterministic correctness reference. Stop that run if available RAM or swap thresholds are breached. Only then assess narrower readahead, lazy loading, KV quantization, or expert residency changes against an unmodified Xing4.0 baseline.
