# Ornith adaptive memory and stream evaluation

These are small functional checks of Ornith-1.5-9B IQ2_M on the two-vCPU host, not a
general reasoning benchmark. The launchers enforce 50% CPU, 4 GiB `MemoryHigh`, 6 GiB
`MemoryMax`, 19 GiB `MemorySwapMax`, and pressure/availability guards. No model weights
were changed.

## Repair memory

The first live slug repair passed isolated tests. Repeating the same objective and
repository snapshot replayed its verified proposal with zero model calls while rerunning
baseline and staged tests. Changing the test file invalidated the proposal cache and
forced a new model request with the verified skill in the prompt.

The first memory-guided request used a 192-token output cap and timed out; its scope
also reached the cgroup pressure guard. The follow-up used a concise repair response
contract and a 160-token cap. It seeded the earlier **live model patch** into a fresh
memory store, reran tests to verify that seed, checked cache replay, then changed a test
and made one new live model request. The new patch passed the changed staged tests.

| Phase | Model calls | Verification runs | Result |
| --- | ---: | ---: | --- |
| Verified seed | 0 in this run | 2 | Passed; source unchanged |
| Identical replay | 0 | 2 | Passed; context index reused |
| Changed test, memory-guided | 1 | 2 | Passed; 407 prompt tokens, 105 output tokens |

The memory-guided request took 295.68 seconds, with first stream content at 94.38
seconds and server-reported generation at 0.517 tokens/s. Peak model RSS was 4086.3 MiB.
The seed's prior live generation and the guided call used different prompt conditions,
so this does not establish that memory improves accuracy or raw model speed. It shows
that evidence-gated reuse and a memory-guided fresh repair both function under the cap.

## Stream and reasoning

The stream probe selected two source chunks and exposed `reasoning_content` separately
from `content`. Its first bounded answer chose the wrong workflow action. The checker
fed back that isolation and rollback were complete; the second answer was
`{"answer":"restart the service"}`. The initial scorer mistakenly required a capital
`R`; [the rescored artifact](../artifacts/ornith-stream-probe-feedback-rescored.json)
accepts the case-equivalent answer while preserving the first failed attempt. The two
server-reported generation rates were 0.513 and 0.521 tokens/s. A longer 96-token probe
ran at 0.361 tokens/s while another client shared the same CPU quota and ended with
truncated JSON. The shorter reasoning budget and checker feedback avoided that waste.

The trace shows text the model emitted as reasoning, not its hidden internal state.
Context lookup remains a fast SQLite search. Within a research round, the backend can
stream model deltas; after a cited note is checked, the agent emits that note and follows
the next source query. The current llama.cpp request cannot ingest a newly retrieved
chunk while the same generation is in progress, so this trace improves visibility and
time to first visible result, not tokens per second.

Run the probe with `bash scripts/run_ornith_stream_probe.sh`. The live repair evaluator
is `bash scripts/run_ornith_adaptive_eval.sh --seed-from artifacts/ornith-adaptive-eval.json`.
Raw results are in `artifacts/ornith-adaptive-eval.json`,
`artifacts/ornith-adaptive-guided-seeded.json`, and the stream artifacts. The scorer can
be replayed without a model using `python3 scripts/rescore_ornith_stream_probe.py INPUT OUTPUT`.
