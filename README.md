# Xing-Specialist

The current two-model experiment uses VibeThinker-1.5B for planning and
DeepSeek-R1-Distill-Qwen-1.5B for bounded execution. See the
[router protocol](docs/two-model-router.md) and run the verified local suite with
`bash scripts/run_dual_pipeline.sh --suite`. The Ornith instructions and results
below document the previous 9B baseline.
The corrected local suite passed 4/9 cases; a small-data LoRA pilot is
documented separately in [the training report](docs/lora-pilot.md) and is not
enabled in the router.
The [independent candidate evaluation](docs/small-model-candidate-evaluation.md)
compares Qwen3, Granite, and Gemma against those baselines; Youtu currently
cannot run on this host's llama.cpp CPU backend.

For a new host or coding-agent session, start with [the handoff](docs/new-instance-handoff.md).

A bounded software-repair agent with a model-independent local HTTP backend, isolated test execution, and evidence-backed SQLite memory. The temporary Qwen coding model remains available; Ornith-1.5-9B is under bounded evaluation on this 8 GB host. The Xing4.0 GGUF was removed after its load attempts failed and can be reacquired later.

## Run

From this directory, start the resource-capped Ornith server in one terminal:

```bash
bash scripts/serve_ornith_model.sh
```

In another terminal, stage a repair of a repository you choose:

```bash
scripts/run_agent.sh \
  --repo examples/repair-demo \
  --objective 'Fix add so the test passes' \
  --allow-path calc.py \
  --test 'python3 -m unittest discover -s tests' \
  --context 1024 --max-output 192 --timeout 420
```

The default leaves the source unchanged and returns a tested proposal. Add `--apply` to promote only when a failing baseline test passes after the staged change; the agent runs the test again after promotion. Test files cannot be automatically promoted. Only explicitly approved files and test commands are accepted. State defaults to `~/.local/share/xing-specialist/memory.sqlite3`. Use `--backend-url`, `--model-id`, `--context`, `--max-output`, and `--timeout` to change models through the same API. The backend endpoint must be local HTTP in this initial release.

For large source material, repeat `--context-source path/to/file.txt` with repository-relative text files. The agent streams those files into a local SQLite index, can queue follow-up searches with `--context-rounds` (1-4), and routes matching chunks plus short working notes into its bounded model request. This is disk-backed retrieval over a large corpus, not a native million-token attention window. The index is stored at `~/.local/share/xing-specialist/context.sqlite3` by default.

Add `--stream-trace` to emit JSON lines for selected context chunks, cited research notes, and model content or reasoning deltas as the server sends them. The agent emits a cited note before following its next source query; citation alone does not verify the note. The final result includes measured generation tokens per second when the server reports timing data. Use `--thinking-budget 16` to request a short model-provided reasoning stream when the server supports it. A reasoning delta is model-generated text, not access to hidden model state. The bounded Ornith stream probe is `bash scripts/run_ornith_stream_probe.sh`; it records the trace in `artifacts/ornith-stream-probe.json` and requires a complete, checked answer.

[Selective attention and self-correction](docs/adaptive-memory.md) adds bounded memory for recurring questions and verified procedures. An unchanged, previously verified repair can skip another model call, but approved tests still run. The agent does not update GGUF weights online.

The [bounded Ornith adaptive and stream evaluation](docs/ornith-adaptive-stream-eval.md) records a tested cache replay, one successful memory-guided fresh repair, server-reported token rates, and a reasoning task that needed checker feedback. These are functional checks, not evidence of a general intelligence gain.

For deeper repair analysis, add `--reasoning-workers 2` (1-3). The agent asks the model to split the objective into up to three testable units, analyzes them through independent bounded calls (concurrently when workers exceed one), and reviews the findings for contradictions before proposing a patch. Unit progress and the review are recorded in SQLite, and stage/unit status appears in the CLI JSON result. An analyzed unit is not a verified result; only approved sandboxed tests can verify the repair. This adds model calls and may be slower on a memory-limited local server. The default is 0 to retain the measured single-pass path.

Ornith is the target model. The legacy temporary Qwen launcher remains single-slot and is not part of the Ornith concurrency result. The shared `llama.cpp` adapter now accepts a bounded slot count. The [two-slot Ornith probe](artifacts/ornith-parallel-probe.json) returned both expected answers in 17.99 seconds with 1,024 tokens per slot, 4,087.6 MiB peak RSS, and no memory guard breach. This establishes short-request concurrency; it does not establish throughput for longer reasoning requests. Tasks remain bounded by `--max-attempts` and model timeouts; interrupted tasks are recorded in SQLite but cannot yet resume from a checkpoint.

For general research, run `PYTHONPATH=src python3 -m xing_specialist.research_cli --objective 'your question' --workspace . --local-source docs/example.md` or supply `--seed-url https://example.org/` and `--max-pages 4`. Omit sources to search the saved library without recrawling. The collector follows relevant same-host HTTPS links within the page budget, stores local references under the workspace and web references globally, and makes both searchable by the repair agent. Add `--extract` while the Ornith server is running to ask it for one useful quote classified as a claim or procedure. Extraction uses a short relevant passage and a 64-token output budget by default. Quotes must match captured source text apart from whitespace, and the stored quote preserves the source's actual line breaks. This establishes provenance, not independent truth. Changed sources invalidate their old findings. Web collection accepts public HTTPS text/HTML only, with a 1 MiB page limit. See [research verification](docs/research-verification.md) for the precise status of each workflow.

Prompt evaluation on Ornith-1.5-9B must use `bash scripts/run_ornith_eval.sh --case boundary` or `bash scripts/run_ornith_eval.sh --quiz`. The launcher places the server and evaluator in a systemd scope with a 50% CPU quota, 4 GiB soft and 6 GiB hard RAM limits, a 19 GiB swap ceiling, low I/O weight, and a 10-minute deadline. The evaluator refuses to start outside that scope. Under these limits, the slug repair passed in 388 seconds; a boundary repair timed out after 420 seconds, and a constrained short coding quiz passed 1 of 3 questions. Ornith IQ2_M in non-thinking mode is not yet a reliable agent backend on this host. The optional small-model smoke test is `python3 scripts/evaluate_prompt.py`; it does not represent Ornith quality. A Rust streaming retrieval candidate lives in `experiments/rust-context-probe`; it is not part of the agent. Build and test it with `cargo test --manifest-path experiments/rust-context-probe/Cargo.toml`, then compare it with SQLite using `cargo build --release --manifest-path experiments/rust-context-probe/Cargo.toml` and `python3 scripts/compare_context_routers.py`.

The [verified reasoning experiment](docs/verified-reasoning-experiment.md) covers quantitative reasoning, logic, planning, and coding. `scripts/prepare_verified_tasks.py` compiles curated, checked examples into family-separated train/eval data for a small-data LoRA/QLoRA and GGUF comparison. The nine seed cases are a pipeline smoke test, not evidence of a quality gain; no Ornith weights have been modified.

The [Ornith reasoning stress report](docs/ornith-reasoning-stress.md) records bounded quantitative, logic, and multi-step planning runs with executable checkers and feedback retries. Run a single challenge through `bash scripts/run_ornith_stress.sh --case staggered-pumps --tool-assisted`; the launcher enforces the same CPU and memory scope described in the report. `scripts/rescore_ornith_stress.py` can replay saved responses through the current checker without loading the model.

The [bytecode blind-spot probe](docs/ornith-bytecode-analysis.md) checks whether Ornith identifies boundary guards disabled by `python -O`. Source-only and raw-bytecode prompts missed the original cause; a disassembler-generated delta led to a correct explanation. The evidence-and-feedback mode corrected the original case semantically, but Ornith still missed a held-out `__debug__` case after feedback. The tool independently verified both flaws. Run it with `bash scripts/run_ornith_bytecode_probe.sh --mode feedback` under the standard resource envelope.

The [multi-file investigation probe](docs/ornith-investigation.md) tests whether Ornith explores a regression, records predictions, revises its search path, and proposes an edit that passes isolated tests. Run `bash scripts/run_ornith_investigation.sh --case tenant_reports` or `--case discount_shipping`; each run saves its action trace and separate exploration metrics.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

See [Stage 5 details](docs/stage5-agent.md), the [model manifest](configs/model-manifest.json), and earlier [feasibility research](docs/memory_estimates.md). Xing4.0 would require a separately bounded short-context test if reacquired; its 11.8 GB GGUF exceeded physical RAM here.
