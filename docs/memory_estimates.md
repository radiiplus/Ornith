# Xing4.0 memory and disk estimates

Checked 2026-10-29 UTC. File sizes below are exact bytes from the publisher's [Hugging Face GGUF API listing](https://huggingface.co/api/models/XingChen-AGI/Xing4.0-29B-A4B-GGUF?blobs=true) at revision `195d1bd04080d73bdead6f40d8d7d1a8f43d5050`. MiB/GiB figures are binary. Resident and throughput figures are planning estimates, not benchmarks.

| Candidate | File size | MiB/GiB | Initial suitability |
| --- | ---: | ---: | --- |
| BF16 safetensors | 62,438,573,536 B | 58.15 GiB | No: exceeds RAM + swap even before overhead |
| FP8 repository | about 33.2 GB | about 30.9 GiB | No: GPU-oriented and exceeds RAM + swap with overhead |
| GGUF IQ2_M-slim | 11,800,813,760 B | 10.99 GiB | Smallest publisher option; quality and CPU compatibility unmeasured |
| GGUF IQ3_XXS | 13,989,927,648 B | 13.03 GiB | More weight pressure |
| GGUF IQ3_M | 15,446,399,712 B | 14.39 GiB | More weight pressure |
| GGUF IQ4_XS | 18,272,862,944 B | 17.02 GiB | Severe pressure |
| GGUF IQ4_NL | 20,104,012,544 B | 18.72 GiB | Severe pressure; common published variant |
| GGUF Q8_0 | 34,164,155,104 B | 31.82 GiB | No |

The current [Stage 1 budget](../configs/memory-budget.json) reserves 2.0 GiB OS/services and 1.0 GiB runtime/KV/temporaries, leaving 4.5 GiB initially for resident model pages. Thus even IQ2_M-slim has a **6.5 GiB gap** between full-file residency and the initial model-page budget. Memory mapping allows cold clean pages to stay only on SSD and hot pages to occupy page cache/RSS, but access to many experts across requests may repeatedly fault them in. A plausible process peak also needs 0.5-2+ GiB of temporary graph/tensor memory (unmeasured) and cache growth. A swap allocation is not equivalent to extra RAM. A 20 GiB swap file can absorb anonymous memory pressure but cannot make a cold, mmap-backed 11 GB model behave as though resident.

Theoretical address-space floor is approximately the mapped file plus runtime and KV allocations; **there is no reliable minimum resident set without measuring which expert pages stay hot**. For planning, near-full residency of the weight file plus 2 GiB OS reserve, 1 GiB runtime reserve, and 90 MiB compressed F16 K cache at 2K context gives these host-memory envelopes:

| Candidate | Near-full-residency host RAM envelope | Gap above 7.60 GiB physical RAM |
| --- | ---: | ---: |
| IQ2_M-slim | 14.08 GiB | 6.48 GiB |
| IQ3_XXS | 16.12 GiB | 8.52 GiB |
| IQ3_M | 17.48 GiB | 9.88 GiB |
| IQ4_XS | 20.11 GiB | 12.51 GiB |
| IQ4_NL | 21.81 GiB | 14.21 GiB |
| BF16 | at least 62.7 GiB using its expanded 2K KV cache | over 55 GiB |
| FP8 | roughly 34 GiB before engine-specific GPU buffers | over 26 GiB |

The gaps are exposure to file-page eviction and possibly anonymous-memory swap, **not a forecast of swap usage**. The 1 GiB runtime reserve may be too small, especially during prefill; the 0.5-2+ GiB temporary estimate is the risk range to measure, not an additional certified allocation. If model pages cannot remain hot, the practical requirement is more RAM or a smaller model, not merely more swap.

## KV-cache models

For the Xing4.0 `llama.cpp` port's compressed MLA path, logical K is 512 latent + 64 RoPE = 576 elements per token/layer. At F16 and 40 layers, **45 KiB/token** (K only) is a lower bound; if a separate 512-element V buffer is materialized, K+V reaches **85 KiB/token**. Actual allocation, alignment and cache quantization depend on engine settings. The reference Transformers code expands to 32 heads of 192-dimensional K and 128-dimensional V: **800 KiB/token** at BF16 for 40 layers, before allocator overhead.

| Context tokens | Compressed F16 K-only | Compressed F16 K+V | Transformers BF16 expanded K+V |
| ---: | ---: | ---: | ---: |
| 2,048 | 90 MiB | 170 MiB | 1.56 GiB |
| 4,096 | 180 MiB | 340 MiB | 3.13 GiB |
| 8,192 | 360 MiB | 680 MiB | 6.25 GiB |
| 32,768 | 1.41 GiB | 2.66 GiB | 25 GiB |
| 262,144 | 11.25 GiB | 21.25 GiB | 200 GiB |

These are single-sequence linear estimates, not guaranteed engine allocations. Long context also increases attention work and transient prefill memory. Start at 2K tokens, one sequence, and record actual cache allocation. The advertised 256K context is infeasible on this host.

## Storage and speed

Root has about 77 GiB free now, with swap already allocated. Downloading one 11.8 GB GGUF directly to a persistent model directory leaves roughly 65 GB before caches/logs; a temporary download may briefly double that file's footprint. A BF16-to-GGUF conversion would require 62.4 GB source plus a large intermediate GGUF and target output, so **cannot be safely budgeted on this 100 GiB volume**. Use a publisher GGUF and avoid duplicate Hugging Face cache copies. A single 20.1 GB GGUF is disk-feasible but harder to run in memory.

With two vCPUs, no GPU, and a model larger than RAM, CPU compute plus model-page I/O are likely the dominant bottlenecks. Working hypothesis: an IQ2 CPU run may be below 1 token/s if expert pages thrash, but there is **no measured throughput** and hot-cache behavior could differ. Measure tokens/s, RSS, major faults, swap I/O, and PSI before selecting a serving path; there is no defensible sustained-rate guarantee from file size or 4B active parameters alone.
