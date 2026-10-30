"""Command-line entry point for the engineering agent."""

import argparse
import json
from pathlib import Path
import shlex

from .agent import EngineeringAgent
from .backend import GenerationLimits, OpenAICompatibleBackend
from .context import ContextRouter
from .memory import MemoryStore
from .research import ResearchLibrary
from .tools import IsolatedExecutor


def main():
    parser = argparse.ArgumentParser(description='Run a bounded, evidence-backed engineering task')
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--objective', required=True)
    parser.add_argument('--allow-path', action='append', required=True)
    parser.add_argument('--context-source', action='append', default=[],
                        help='Repository-relative text file to index as disk-backed context; repeat for more files')
    parser.add_argument('--context-rounds', type=int, default=2,
                        help='Maximum routed context searches per task (1-4, default: 2)')
    parser.add_argument('--test', action='append', required=True,
                        help='Approved test command, e.g. "python3 -m unittest discover -s tests"')
    parser.add_argument('--backend-url', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--model-id', default='local')
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--max-output', type=int, default=512)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--reasoning-workers', type=int, default=0, choices=range(4),
                        help='Analyze 1-3 objective units with independent model calls; 0 disables it')
    parser.add_argument('--stream-trace', action='store_true',
                        help='Emit source-chunk lookups and model content/reasoning deltas as JSON lines')
    parser.add_argument('--thinking-budget', type=int, default=0,
                        help='Request up to this many thinking tokens (0-128; server support required)')
    parser.add_argument('--apply', action='store_true', help='Promote a staged patch only after tests pass')
    parser.add_argument('--state-dir', type=Path,
                        default=Path.home() / '.local' / 'share' / 'xing-specialist')
    args = parser.parse_args()
    limits = GenerationLimits(args.context, args.max_output, args.timeout)

    def emit_model(kind, part):
        print(json.dumps({'event': 'model_' + kind, 'text': part}), flush=True)

    def emit_source(source, ordinal, chars):
        print(json.dumps({'event': 'context_chunk', 'source': source,
                          'ordinal': ordinal, 'chars': chars}), flush=True)

    def emit_event(kind, detail):
        print(json.dumps({'event': kind, 'text': detail}), flush=True)

    backend = OpenAICompatibleBackend(args.backend_url, args.model_id,
                                      on_chunk=emit_model if args.stream_trace else None,
                                      thinking_budget=args.thinking_budget)
    store = MemoryStore(args.state_dir / 'memory.sqlite3')
    research = ResearchLibrary(args.state_dir / 'research.sqlite3')
    router = (ContextRouter(args.state_dir / 'context.sqlite3',
                            on_chunk=emit_source if args.stream_trace else None)
              if args.context_source else None)
    try:
        executor = IsolatedExecutor(args.state_dir / 'sandboxes')
        agent = EngineeringAgent(backend, store, executor, limits, args.max_attempts, router,
                                 args.context_rounds, args.reasoning_workers, research,
                                 emit_event if args.stream_trace else None)
        outcome = agent.run(args.objective, args.repo, tuple(args.allow_path),
                            tuple(tuple(shlex.split(value)) for value in args.test), args.apply,
                            tuple(args.context_source))
        print(json.dumps({**outcome.__dict__, 'progress': store.progress(outcome.task_id),
                          'model_metrics': backend.last_metrics}, indent=2))
        return 0 if outcome.status in ('verified', 'needs_review') else 1
    finally:
        if router:
            router.close()
        research.close()
        store.close()


if __name__ == '__main__':
    raise SystemExit(main())
