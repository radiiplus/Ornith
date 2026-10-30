# Inference compatibility

Checked 2026-10-29 UTC. Classification distinguishes architecture support from viability on this 2-vCPU, 7.60-GiB-RAM, GPU-less host. No engine was installed or tested locally.

| Option | Classification | Basis |
| --- | --- | --- |
| Stock `llama.cpp` master | **Unsupported** | [Xing4.0 PR #29012](https://github.com/ggml-org/llama.cpp/pull/29012) is open and unmerged (API check 2026-10-29); generic GGUF support is insufficient. |
| Pinned Xing4.0 `llama.cpp` CPU port + publisher GGUF | **Plausible but unverified** | PR implements model loading, conversion, mHC/MLA/MoE graph and CPU backend; [publisher GGUFs](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-GGUF/tree/main) demonstrate conversion is possible. No build or CPU inference on this host yet. |
| Publisher Transformers custom code + BF16 | **Impractical on target hardware** | Architecture path is published, but 62.44 GB weights plus expanded cache and runtime are far beyond 7.60 GiB RAM and 20 GiB swap. |
| FP8 checkpoint through GPU engine | **Impractical on target hardware** | [FP8 repo](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B-FP8/tree/main) is about 33.2 GB and host has no GPU. |
| vLLM + publisher [Xing4 plugin](https://github.com/XingChen-AGI/xing4-vllm-plugin) | **Confirmed compatible** for architecture; **impractical on target hardware** | Publisher ships a model, parser and MLA/MoE plugin; its guide uses GPUs and BF16 mHC. No local execution. Plain vLLM without plugin is unverified. |
| SGLang + publisher [Xing4 plugin](https://github.com/XingChen-AGI/xing4-sglang-plugin) | **Confirmed compatible** for architecture; **impractical on target hardware** | Publisher plugin targets SGLang 0.5.x and GPU MLA/MoE runtimes. No local execution. Plain SGLang without plugin is unverified. |
| KTransformers / TensorRT-LLM | **Impractical on target hardware** | [Publisher guide](https://github.com/XingChen-AGI/Xing4.0-29B-A4B) uses heterogeneous CPU/GPU or NVIDIA GPU examples; related support PRs are pending. |

The upstream [llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/29012) reports CPU backend tests and GGUF conversion. The publisher hosts several GGUF variants, so conversion can retain the ordinary generation architecture in an adapted engine. This does **not** prove a stock GGUF loader preserves mHC/MLA/MoE semantics, that all quantizations preserve quality, or that the MTP head is usable for speculative decoding. Pin a known Xing4.0 branch and compare deterministic output against the reference before claiming parity. A generic `llama.cpp` or vLLM install must not be treated as compatible merely because Hugging Face displays generic usage snippets.

**Recommended next experiment after review:** install a minimal C++ toolchain and compile the pinned Xing4.0 CPU port, then run its model/operator tests without Xing4.0 weights. Validate the host inference and monitoring pipeline with the publisher's [Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) (1,117,320,768 bytes) before considering one Xing4.0 quantized file. Xing4.0 remains the intended target. No engine installation, model download, or model-specific runtime change was made in this stage.
