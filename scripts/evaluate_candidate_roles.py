#!/usr/bin/env python3
"""Measure standalone objective fidelity and workflow action selection."""

import argparse
from collections import deque
import json
from pathlib import Path
import signal
import time

from benchmark_inference import LlamaCppAdapter, Monitor, free_port, preflight
from evaluate_ornith_prompt import ROOT, SERVER
from evaluate_standalone_agent import CASES as REPAIR_CASES
from prepare_verified_repairs import check_plan, load_cases
from stress_ornith_reasoning import CHALLENGES, SEED
from xing_specialist.backend import GenerationLimits, Message, OpenAICompatibleBackend
from xing_specialist.dual_router import PlanProgress, SCHEMAS, objective_gate
from xing_specialist.envelope import require_small_model_envelope


GATE_IDS = ('six-fact-implication', 'converse-fallacy')
WORKFLOW_IDS = ('incident-deployment', 'incident-workflow')


def shortest_plan_length(verifier):
    initial = frozenset(verifier['initial'])
    goal = set(verifier['goal'])
    queue = deque([(initial, 0)])
    seen = {initial}
    while queue and len(seen) < 4096:
        state, depth = queue.popleft()
        if goal <= state:
            return depth
        for action in verifier['actions'].values():
            if not set(action['requires']) <= state:
                continue
            following = frozenset((set(state) - set(action['removes'])) |
                                  set(action['adds']))
            if following not in seen:
                seen.add(following)
                queue.append((following, depth + 1))
    raise ValueError('Verified workflow has no bounded solution')


def gate_prompt(objective):
    return ('Restate this objective before any action as JSON with exactly objective, '
            'subgoals (one or two short strings), assumptions (empty array unless '
            'the original states one), done_when (one short string). Preserve the '
            'question and constraints. Do not invent requirements. done_when must '
            'describe the original success condition, never a date or duration. '
            'Original: ' + objective)


def deadline_reached(_signum, _frame):
    raise TimeoutError('Candidate role probe deadline reached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--chat-template-kwargs', type=str)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_small_model_envelope()
    template_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    if template_kwargs is not None and (set(template_kwargs) != {'enable_thinking'} or
                                        type(template_kwargs['enable_thinking']) is not bool):
        parser.error('chat-template-kwargs must contain only boolean enable_thinking')
    signal.signal(signal.SIGTERM, deadline_reached)
    model = args.model_path.resolve(strict=True)
    preflight(model, 0)
    cases = {row['id']: row for row in load_cases(CHALLENGES) + load_cases(SEED)
             if row.get('verifier')}
    gate_cases = [(case_id, cases[case_id]['question']) for case_id in GATE_IDS]
    gate_cases.append(('tenant_reports', REPAIR_CASES['tenant_reports']['objective']))
    port = free_port()
    engine = LlamaCppAdapter(SERVER, model, 2048, 1, port, 'mmap',
                             batch_size=128, ubatch_size=64, no_warmup=True,
                             request_timeout=180, fit='on')
    engine.command += ['--reasoning', 'off']
    if args.model_name.startswith('DeepSeek-R1-'):
        engine.command += ['--reasoning-budget', '0']
    if template_kwargs is not None:
        engine.command += ['--chat-template-kwargs', json.dumps(template_kwargs)]
    result = {'model': args.model_name, 'model_path': str(model),
              'chat_template_kwargs': template_kwargs, 'cpu_quota': '50%',
              'memory_high': '4G', 'memory_max': '6G', 'swap_max': '19G',
              'objective_gates': [], 'workflows': [], 'status': 'running'}
    monitor = None
    started = time.monotonic()

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    def ask(prompt, schema, max_tokens):
        backend = OpenAICompatibleBackend(
            f'http://127.0.0.1:{port}/v1', 'local',
            response_format={'type': 'json_schema', 'json_schema': {'schema': schema}})
        answer = backend.generate((Message('user', prompt),),
                                  GenerationLimits(2048, max_tokens, 180))
        if monitor.breach_reason:
            raise RuntimeError(monitor.breach_reason)
        return answer, backend.last_metrics

    try:
        engine.start()
        monitor = Monitor(engine.process.pid, 2304 * 1024 ** 2,
                          1536 * 1024 ** 2, 19 * 1024 ** 3,
                          engine.process.terminate, 5.0)
        monitor.start()
        engine.wait_ready(90)
        for case_id, objective in gate_cases:
            record = {'case': case_id, 'format_valid': False, 'objective_fidelity': False}
            try:
                answer, metrics = ask(gate_prompt(objective), SCHEMAS['objective_gate'], 192)
                record.update(response=answer, metrics=metrics)
                item = json.loads(answer)
                record['format_valid'] = isinstance(item, dict) and set(item) == {
                    'objective', 'subgoals', 'assumptions', 'done_when'}
                objective_gate(objective, item)
                record['objective_fidelity'] = True
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and 'empty assistant content' not in str(exc):
                    raise
                record['feedback'] = str(exc)[:240]
            result['objective_gates'].append(record)
            save()
            print(json.dumps({'case': case_id, 'objective_fidelity':
                              record['objective_fidelity']}), flush=True)

        for case_id in WORKFLOW_IDS:
            row = cases[case_id]
            verifier = row['verifier']
            progress = PlanProgress(verifier)
            shortest = shortest_plan_length(verifier)
            record = {'case': case_id, 'actions': [], 'turns': [],
                      'shortest_actions': shortest, 'passed': False}
            for _ in range(len(verifier['actions']) + 2):
                available = progress.available()
                if not available:
                    record['feedback'] = 'No available action makes progress'
                    break
                schema = {'type': 'object', 'properties': {
                    'action': {'type': 'string', 'enum': available}},
                    'required': ['action'], 'additionalProperties': False}
                prompt = ('Select one available action toward the original objective. '
                          'Return JSON with only action. Original objective: ' + row['question']
                          + '\nAction graph: ' + json.dumps(verifier['actions'])
                          + '\nCurrent state: ' + json.dumps(sorted(progress.state))
                          + '\nGoals: ' + json.dumps(sorted(progress.goal))
                          + '\nAvailable actions: ' + json.dumps(available)
                          + '\nHistory: ' + json.dumps(progress.history))
                turn = {'available': available, 'format_valid': False}
                try:
                    answer, metrics = ask(prompt, schema, 64)
                    turn.update(response=answer, metrics=metrics)
                    item = json.loads(answer)
                    turn['format_valid'] = isinstance(item, dict) and set(item) == {'action'}
                    action = item['action']
                    reached = progress.advance(action)
                    record['actions'].append(action)
                    if reached:
                        check_plan(verifier, progress.history)
                        record['passed'] = True
                except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                    if isinstance(exc, RuntimeError) and 'empty assistant content' not in str(exc):
                        raise
                    turn['feedback'] = str(exc)[:240]
                record['turns'].append(turn)
                if record['passed'] or 'feedback' in turn:
                    break
            record['extra_actions'] = (len(record['actions']) - shortest
                                       if record['passed'] else None)
            result['workflows'].append(record)
            save()
            print(json.dumps({'case': case_id, 'passed': record['passed'],
                              'actions': len(record['actions'])}), flush=True)
        result['status'] = 'completed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = str(exc)[:300]
    finally:
        if monitor:
            monitor.stop()
            result['peak_rss_mib'] = round(max((sample['rss_high_water_bytes']
                for sample in monitor.samples), default=0) / 1024 ** 2, 1)
        engine.stop()
        result['wall_seconds'] = round(time.monotonic() - started, 2)
        save()
    print(json.dumps({'status': result['status'],
                      'wall_seconds': result['wall_seconds'],
                      'peak_rss_mib': result.get('peak_rss_mib')}), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
