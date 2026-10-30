# Selective attention and self-correction

The agent keeps a small SQLite attention store alongside its full audit log. This changes
the inference workflow; it does not modify Ornith's GGUF weights or claim that the model
learns from a prompt by itself.

## What is retained

- A failed proposal or test produces a pending gap. A matching gap becomes an open
  question only after it recurs. Research questions generated while exploring selected
  local sources use the same rule. They are labeled as unanswered, not as facts.
- A procedure becomes active only after the task had a failing baseline and the staged
  change passed the approved tests. The record links to the passing test result.
- At most three relevant active procedures or open gaps are inserted into a repair
  prompt. The store is capped at 128 attention items; verified proposal replay is
  capped at 64 entries. Relevance uses lexical overlap and bounded salience.
- A corrected, verified procedure for the same objective replaces its body and
  increments its version. Failed replay reduces its salience and disables the cached
  proposal. A procedure is retired after two failed replays.

## Avoiding repeated work

The agent hashes the eligible repository files, objective, approved paths and test
commands before caching a verified proposal. On an identical snapshot it can replay
that proposal without another model call. It still runs baseline and staged tests.
Changing a source file, test, command, or objective changes the key. Oversized or
unreadable repositories skip replay. Source context files are indexed once per content
hash and chunk configuration, so unchanged material does not churn the SQLite FTS index.

This is conservative reuse of an observed result, not proof that a procedure works in
all cases. Passing tests can be incomplete. Open research questions are not promoted
to verified procedures without independent checks. The separate research collector now
captures selected local files and bounded public HTTPS pages. It stores quote-backed
claims and procedures as source-checked candidates, not established facts, and does not
update model weights. The separate Ornith reasoning stress harness tests arithmetic,
logic, and planning with exact checkers and should remain held out from the memory store
when measuring generalization.

Run the focused checks with:

```bash
PYTHONPATH=src python3 -m unittest tests.test_adaptive tests.test_agent tests.test_context -v
```

The design follows the feedback and episodic-memory direction in
[Reflexion](https://arxiv.org/abs/2303.11366), the verified skill-library idea in
[Voyager](https://arxiv.org/abs/2305.16291), and the selective retrieval pattern in
[Generative Agents](https://arxiv.org/abs/2304.03442). Those papers motivate the
workflow; they are not evidence of a quality improvement for this Ornith configuration.
