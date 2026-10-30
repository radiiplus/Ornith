#!/usr/bin/env python3
"""Run the bounded two-model router and its verified local evaluation cases."""

import argparse
import json
from pathlib import Path
import shlex
import signal
import sys
import time

from benchmark_inference import LlamaCppAdapter, Monitor, free_port, preflight
from evaluate_ornith_prompt import ROOT, SERVER
from prepare_verified_repairs import load_cases
from stress_ornith_reasoning import (CHALLENGES, SEED, arithmetic_value,
                                     normalize_answer, score_answer)
from xing_specialist.backend import GenerationLimits, OpenAICompatibleBackend
from xing_specialist.dual_router import DualModelRouter
from xing_specialist.envelope import require_small_model_envelope
from xing_specialist.memory import MemoryStore
from xing_specialist.research import ResearchLibrary
from xing_specialist.tools import IsolatedExecutor


MODELS = {
    'reasoning': ROOT / 'models/vibethinker-1.5b-q4_k_m.gguf',
    'execution': ROOT / 'models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf',
}
STOP_REQUESTED = False


def deadline_reached(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    raise TimeoutError('Two-model run deadline reached')


class SwappingModelProvider:
    def __init__(self, models=None, chat_template_kwargs=None):
        self.role = None
        self.models = models or MODELS
        self.chat_template_kwargs = chat_template_kwargs
        self.engine = None
        self.monitor = None
        self.port = None
        self.events = []
        self.peak_rss_mib = 0

    def close(self):
        if self.monitor:
            self.monitor.stop()
            if self.monitor.samples:
                self.peak_rss_mib = max(self.peak_rss_mib, max(
                    sample['rss_high_water_bytes'] for sample in self.monitor.samples) / 1024 ** 2)
            self.monitor = None
        if self.engine:
            self.engine.stop()
            self.engine = None
        self.role = None

    def _load(self, role):
        if role == self.role:
            return 0.0
        self.close()
        model = self.models[role]
        preflight(model, 0)
        self.port = free_port()
        self.engine = LlamaCppAdapter(SERVER, model, 2048, 1, self.port, 'mmap',
                                      batch_size=128, ubatch_size=64, no_warmup=True,
                                      request_timeout=180, fit='on')
        self.engine.command += ['--reasoning', 'off']
        if self.chat_template_kwargs is not None:
            self.engine.command += ['--chat-template-kwargs',
                                    json.dumps(self.chat_template_kwargs)]
        if role == 'execution':
            self.engine.command += ['--reasoning-budget', '0']
        started = time.monotonic()
        try:
            self.engine.start()
            self.monitor = Monitor(self.engine.process.pid, 2304 * 1024 ** 2,
                                   1536 * 1024 ** 2, 19 * 1024 ** 3,
                                   self.engine.process.terminate, 5.0)
            self.monitor.start()
            self.engine.wait_ready(90)
            if self.monitor.breach_reason:
                raise RuntimeError(self.monitor.breach_reason)
            self.role = role
            return time.monotonic() - started
        except Exception:
            self.close()
            raise

    def generate(self, role, messages, limits, schema=None):
        load_seconds = self._load(role)
        backend = OpenAICompatibleBackend(
            f'http://127.0.0.1:{self.port}/v1', 'local',
            response_format={'type': 'json_schema', 'json_schema': {'schema': schema}}
            if schema else None)
        try:
            answer = backend.generate(messages, limits)
        except Exception as exc:
            raise RuntimeError(
                f'{role} backend request failed: {type(exc).__name__}: {exc}; '
                f'server_exit={self.engine.process.poll()}; '
                f'memory_guard={self.monitor.breach_reason}') from exc
        if self.monitor.breach_reason:
            raise RuntimeError(self.monitor.breach_reason)
        metrics = {'load_seconds': round(load_seconds, 2), **backend.last_metrics}
        self.events.append({'role': role, **metrics})
        return answer, metrics


def corpus_files(directory):
    if not directory or not directory.is_dir():
        return ()
    return tuple(sorted(path for path in directory.rglob('*') if path.is_file()
                        and (path.name.startswith('areas') or
                             'representation' in path.name.lower())))[:40]


def verified_context(row):
    verifier = row['verifier']
    if verifier['kind'] == 'logic':
        keys = ('variables', 'premises', 'conclusion')
    elif verifier['kind'] == 'plan':
        keys = ('initial', 'goal', 'actions')
    elif verifier['kind'] == 'arithmetic':
        keys = ()
    else:
        raise ValueError('Unsupported verified task kind')
    return {key: verifier[key] for key in keys}


def check_verified_answer(row, answer):
    if row['verifier']['kind'] == 'arithmetic':
        answer = str(arithmetic_value(answer))
    normalized, _ = normalize_answer(row, answer)
    score_answer(row, normalized)


def main():
    signal.signal(signal.SIGTERM, deadline_reached)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='store_true')
    parser.add_argument('--suite-case', help='Run one verified suite case while tuning')
    parser.add_argument('--objective')
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--allow-path', action='append', default=[])
    parser.add_argument('--test', action='append', default=[])
    parser.add_argument('--corpus-dir', type=Path, default=ROOT / 'corpus/hunt')
    parser.add_argument('--research-seed-url', action='append', default=[])
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--prompt-style', choices=('system', 'user-only', 'role-aware'),
                        default='system')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/dual-model-eval.json')
    parser.add_argument('--state-dir', type=Path)
    args = parser.parse_args()
    require_small_model_envelope()
    if args.suite == bool(args.objective):
        parser.error('Choose --suite or --objective with --repo, --allow-path, and --test')
    if args.suite_case and not args.suite:
        parser.error('--suite-case requires --suite')
    for model in MODELS.values():
        preflight(model, 0)
    state = args.state_dir or Path.home() / '.local/share/xing-specialist'
    store = MemoryStore(state / 'memory.sqlite3')
    research = ResearchLibrary(state / 'research.sqlite3')
    executor = IsolatedExecutor(state / 'sandboxes', timeout_seconds=20, memory_mib=256)
    provider = SwappingModelProvider()
    router = DualModelRouter(provider, store, executor,
                             GenerationLimits(2048, 192, 180), args.max_attempts, 1,
                             corpus_files(args.corpus_dir), research,
                             tuple(args.research_seed_url), args.prompt_style)
    result = {'models': {'reasoning': 'VibeThinker-1.5B Q4_K_M',
                         'execution': 'DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M'},
              'cpu_quota': '100%', 'memory_high': '4G', 'memory_max': '6G',
              'swap_max': '19G', 'cases': [], 'status': 'running'}
    result['prompt_style'] = args.prompt_style
    started = time.monotonic()

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')

    def run_one(name, objective, repo, paths, tests, checker=None, answer_format=None,
                task_context=None):
        case_start = time.monotonic()
        event_start = len(provider.events)
        try:
            outcome = router.run(objective, repo, paths, tests, checker,
                                 answer_format, task_context)
            if STOP_REQUESTED:
                raise TimeoutError('Two-model run deadline reached')
            row = {**outcome.__dict__, 'case': name, 'wall_seconds': round(time.monotonic() - case_start, 2),
                   'passed': outcome.status == 'verified'}
        except Exception as exc:
            if STOP_REQUESTED:
                raise
            row = {'case': name, 'status': 'failed', 'passed': False,
                   'error': str(exc), 'wall_seconds': round(time.monotonic() - case_start, 2)}
        events = provider.events[event_start:]
        tokens = sum(event.get('completion_tokens') or 0 for event in events)
        row['generated_tokens'] = tokens
        row['effective_tps'] = round(tokens / row['wall_seconds'], 3) if row['wall_seconds'] else None
        row['model_events'] = events
        active_peak = (max(sample['rss_high_water_bytes'] for sample in provider.monitor.samples) /
                       1024 ** 2 if provider.monitor and provider.monitor.samples else 0)
        row['peak_rss_mib'] = round(max(provider.peak_rss_mib, active_peak), 1)
        result['cases'].append(row)
        save()
        print(json.dumps({'case': name, 'status': row['status'],
                          'wall_seconds': row['wall_seconds']}), flush=True)

    try:
        if args.suite:
            demo = ROOT / 'examples/repair-demo'
            tests = (('python3', '-m', 'unittest', 'discover', '-s', 'tests'),)
            rows = load_cases(CHALLENGES) + [row for row in load_cases(SEED)
                                            if row.get('kind') == 'reasoning'
                                            and row.get('split') == 'eval']
            available = {row['id'] for row in rows} | {'repair-demo', 'tenant_reports'}
            if args.suite_case and args.suite_case not in available:
                parser.error('Unknown --suite-case')
            for row in rows:
                if args.suite_case and args.suite_case != row['id']:
                    continue
                def check(answer, item=row):
                    check_verified_answer(item, answer)
                verifier = row['verifier']
                task_context = verified_context(row)
                run_one(row['id'], row['question'], demo, ('calc.py',), tests, check,
                        verifier['kind'], task_context)
            if args.suite_case in (None, 'repair-demo'):
                run_one('repair-demo', 'Fix add so the test passes', demo, ('calc.py',), tests)
            tenant = ROOT / 'examples/investigation/tenant_reports'
            if args.suite_case in (None, 'tenant_reports'):
                run_one('tenant_reports',
                        'A user authorized for two tenants sees the first tenant report after switching to the second. Find and repair the cause.',
                        tenant, ('reporting/service.py', 'reporting/cache.py', 'reporting/auth.py'), tests)
        else:
            if not args.repo or not args.allow_path or not args.test:
                parser.error('Custom objectives need --repo, --allow-path, and --test')
            run_one('custom', args.objective, args.repo, tuple(args.allow_path),
                    tuple(tuple(shlex.split(command)) for command in args.test))
        result['status'] = 'completed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = str(exc)
    finally:
        provider.close()
        result['peak_rss_mib'] = round(provider.peak_rss_mib, 1)
        result['wall_seconds'] = round(time.monotonic() - started, 2)
        result['model_events'] = provider.events
        save()
        research.close()
        store.close()
    print(json.dumps({'status': result['status'], 'cases': len(result['cases']),
                      'passed': sum(case['passed'] for case in result['cases']),
                      'wall_seconds': result['wall_seconds'],
                      'peak_rss_mib': result['peak_rss_mib']}), flush=True)
    return 0 if result['status'] == 'completed' and all(
        item['passed'] for item in result['cases']) else 1


if __name__ == '__main__':
    sys.exit(main())
