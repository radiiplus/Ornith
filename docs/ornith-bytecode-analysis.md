# Ornith bytecode blind-spot probe

The fixture at `examples/bytecode-boundary-demo/report_reader.py` appears to check that
a resolved report path remains inside a workspace. Its check is an `assert`. Python
removes assertions under `-O`, so the compiled function still calls `read_text` but no
longer calls `is_relative_to`. A local subprocess check confirms the effect: normal
execution rejects `../outside.txt`, while optimized execution reads the marker from a
file outside the workspace. The fixture uses temporary files and no network access.

`scripts/evaluate_ornith_bytecode.py` compares three prompts against the same checked
fixture. It asks Ornith to classify the failure and explain it using source alone,
source plus optimized bytecode, or a deterministic disassembler comparison of normal
and optimized bytecode. The comparison reports the removed `is_relative_to` method and
the remaining `read_text` method. The model does not receive the subprocess result.

| Prompt | Observed classification | Checked result |
| --- | --- | --- |
| Source only | Unsafe, but attributed to generic path traversal/injection | Failed |
| Source and optimized bytecode | Same incorrect cause | Failed |
| Source, optimized bytecode, and method delta | Assert stripped under `python -O`; workspace escape follows | Passed |

The original responses and timings are in `artifacts/ornith-bytecode-probe.json` and
`artifacts/ornith-bytecode-guided.json`. The latter was initially marked failed because
the scorer omitted the word "stripped" from its evidence synonyms. The raw responses
were preserved, and `scripts/rescore_ornith_bytecode.py` wrote separate `-rescored.json`
artifacts. The guided pass took 157 seconds. All runs used Ornith-1.5-9B IQ2_M within
the existing CPU, RAM, swap, and deadline envelope.

Run the local checks with `PYTHONPATH=src python3 -m unittest
tests.test_ornith_bytecode_probe -v`. Run the full model comparison with `bash
scripts/run_ornith_bytecode_probe.sh`, or the guided pass alone with `bash
scripts/run_ornith_bytecode_probe.sh --mode delta --output
artifacts/ornith-bytecode-guided.json`.

This is one intentionally vulnerable Python case. A robust static analyzer can flag
security checks written as assertions, so the result does not show that bytecode is
uniquely capable of finding this flaw. It shows that raw disassembly did not help this
Ornith configuration on its own, while a deterministic bytecode comparison supplied
the missing evidence for a correct explanation.
