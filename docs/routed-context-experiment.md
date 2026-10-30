# Routed context experiment

The local agent can index selected repository text files in SQLite FTS5 and search them in bounded rounds before proposing a code change. `--context-source` selects files, and `--context-rounds` controls the maximum searches (default 2, maximum 4). Each search loads at most a few small chunks into a model request. The model may add a short note citing a retrieved source label and queue up to two follow-up queries. Notes without a retrieved source label are discarded. The final repair request includes the latest retrieved chunks and compact notes; the active model context remains at the configured `--context` size.

This design can store a million-token corpus on disk without allocating a million-token KV cache. It cannot provide exact attention over every token or guarantee that retrieval finds all relevant evidence. FTS5 is lexical, so paraphrases and dependencies spread across distant files may be missed. Notes are model output and need checking against the cited source chunks. The experiment should measure retrieval recall on known facts, repair success on tasks whose evidence spans files, number of model calls, time to answer, and peak memory against the same tasks without routing.

Example from this repository, with the small temporary model running separately:

```bash
scripts/run_agent.sh --repo examples/repair-demo \
  --objective 'Fix add so the test passes' --allow-path calc.py \
  --context-source tests/test_calc.py --context-rounds 2 \
  --test 'python3 -m unittest discover -s tests'
```

Only explicit repository-relative text files are indexed; each file is capped at 128 MiB. The index is scoped to the repository path, and the active run searches only the selected files. Files are streamed in 800-character chunks with 100-character overlap. The prompt builder applies the backend's conservative context budget to every request. No model weights or native attention kernels are modified.

The synthetic scale probe (`python3 scripts/probe_routed_context.py`) indexed one million repeated words plus a unique marker in about 0.6 seconds on this instance. Its 6.2 MB source produced 8,858 chunks and a 10.7 MB SQLite file; the process peaked near 20 MiB RSS and retrieved the marker. This shows that indexing and lexical lookup fit the hardware. It does not measure model reasoning quality, full-context attention, or real repository retrieval recall.

`experiments/rust-context-probe` is a zero-dependency Rust candidate for bounded, streaming lexical retrieval. It reads 800-byte windows with 100-byte overlap, keeps at most six scored chunks, and caps its output. Rust 1.99.1 and Cargo are installed under `~/.cargo/bin`; both Rust unit tests pass. The release build and `scripts/compare_context_routers.py` retrieve the marker from the same 6.2 MB corpus. Across four consecutive runs, SQLite indexing took about 0.54 seconds and its median warm lookup about 0.42 ms; the Rust scanner took about 10.4 ms per query including process startup and no index construction. Typical process peaks were about 21 MiB for the Python/SQLite benchmark and 11 MiB for its Rust child. An earlier run reported a 237 MiB Python peak that has not been explained, so these memory figures are not a hard bound. The scanner reads the source on every query, while SQLite keeps an FTS index for repeated searches. Its byte windows can split UTF-8 characters and it does not apply the agent's redaction or repository path checks, so it must not handle production context as written.
