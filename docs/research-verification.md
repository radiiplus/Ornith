# Research sources and verification

`xing_specialist.research_cli` stores a source with a locator, workspace or global scope,
capture time, content digest, and searchable text. It can follow relevant links on the
seed host within a page budget. The optional model pass proposes claims and reusable
procedures. A finding is accepted only when its quoted words and punctuation occur in
the captured text in order; line wrapping and other whitespace may differ. The stored
quote uses the source's actual text. Changed source content invalidates its old findings.

`source_checked` means the captured source contains the quote. It does not prove that
the source is accurate, current, independent of another source, or sufficient to
support the proposed conclusion. Search results expose capture time so the caller can
decide whether a fresh fetch is needed. Workspace findings stay within that workspace;
global findings can be retrieved by any workspace. The repair agent labels both as
research references and still requires its own approved tests.

Verification depends on the objective:

| Objective | Current independent check | Remaining limit |
| --- | --- | --- |
| Repair | A failing baseline becomes passing under approved isolated tests, then is retested after promotion | Tests may omit behavior |
| Research | Exact quotation from a captured, hashed source | Source truth and freshness need external corroboration |
| Vulnerability discovery | Caller-supplied isolated reproduction can be run as an approved test | No autonomous exploit validator or severity assessment yet |
| New build | Caller-supplied acceptance tests can run in the sandbox | The current agent only edits approved existing files; it cannot create a full project |

No workflow is marked verified merely because the model, a second model call, or a web
page says it succeeded. The research library is an evidence index and candidate
procedure catalog, not an online change to Ornith's parameters.

On this host, a live Ornith extraction over a short passage took about two minutes.
Its quote differed from the Markdown source only by a line break; replaying the saved
model response through the whitespace-aware verifier accepted one finding. This checks
the capture, model-output, and quote-verification path, but it is not a research-quality
benchmark.
