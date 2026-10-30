#!/usr/bin/env python3
"""Benchmark one small model through the existing isolated repair agent."""

import argparse
import json
from pathlib import Path
import signal
import tempfile
import time

from evaluate_ornith_prompt import ROOT
from run_dual_pipeline import MODELS, SwappingModelProvider
from xing_specialist.agent import EngineeringAgent
from xing_specialist.backend import GenerationLimits
from xing_specialist.dual_router import RoleBackend
from xing_specialist.envelope import require_small_model_envelope
from xing_specialist.memory import MemoryStore
from xing_specialist.tools import IsolatedExecutor


CASES = {
    'repair-demo': {
        'objective': 'Fix add so the test passes',
        'repo': ROOT / 'examples/repair-demo', 'paths': ('calc.py',),
    },
    'tenant_reports': {
        'objective': ('A user authorized for two tenants sees the first tenant report '
                      'after switching to the second. Find and repair the cause.'),
        'repo': ROOT / 'examples/investigation/tenant_reports',
        'paths': ('reporting/service.py', 'reporting/cache.py', 'reporting/auth.py'),
    },
}
TESTS = (('python3', '-m', 'unittest', 'discover', '-s', 'tests'),)


def deadline_reached(_signum, _frame):
    raise TimeoutError('Standalone agent deadline reached')


def main():
    signal.signal(signal.SIGTERM, deadline_reached)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=MODELS, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--model-name')
    parser.add_argument('--chat-template-kwargs', type=str)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_small_model_envelope()
    template_kwargs = (json.loads(args.chat_template_kwargs)
                       if args.chat_template_kwargs else None)
    if template_kwargs is not None and (set(template_kwargs) != {'enable_thinking'} or
                                        type(template_kwargs['enable_thinking']) is not bool):
        parser.error('chat-template-kwargs must contain only boolean enable_thinking')
    model_path = args.model_path.resolve(strict=True) if args.model_path else MODELS[args.role]
    case = CASES[args.case]
    state = Path(tempfile.mkdtemp(prefix='xing-standalone-'))
    store = MemoryStore(state / 'memory.sqlite3')
    provider = SwappingModelProvider({args.role: model_path}, template_kwargs)
    started = time.monotonic()
    result = {'model': args.model_name or model_path.name, 'role': args.role,
              'case': args.case, 'chat_template_kwargs': template_kwargs,
              'cpu_quota': '50%', 'memory_high': '4G', 'memory_max': '6G',
              'swap_max': '19G'}
    try:
        executor = IsolatedExecutor(state / 'sandboxes', timeout_seconds=20,
                                    memory_mib=256)
        agent = EngineeringAgent(RoleBackend(provider, args.role, case['paths']), store, executor,
                                 GenerationLimits(2048, 192, 180), max_attempts=2)
        outcome = agent.run(case['objective'], case['repo'], case['paths'], TESTS)
        result.update(outcome.__dict__)
        result['passed'] = (outcome.status == 'needs_review' and
                            outcome.detail == 'Patch tested in staging only')
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='failed', passed=False, error=str(exc))
    finally:
        provider.close()
        store.close()
        result['wall_seconds'] = round(time.monotonic() - started, 2)
        result['peak_rss_mib'] = round(provider.peak_rss_mib, 1)
        result['model_events'] = provider.events
        result['generated_tokens'] = sum(event.get('completion_tokens') or 0
                                         for event in provider.events)
        result['effective_tps'] = (round(result['generated_tokens'] / result['wall_seconds'], 3)
                                   if result['wall_seconds'] else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'case': args.case, 'role': args.role, 'passed': result['passed'],
                      'wall_seconds': result['wall_seconds'],
                      'peak_rss_mib': result['peak_rss_mib']}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
