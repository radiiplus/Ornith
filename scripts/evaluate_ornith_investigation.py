#!/usr/bin/env python3
"""Evaluate bounded, tool-using investigations of multi-file regressions."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
from time import monotonic
from urllib.request import Request, urlopen

from benchmark_inference import LlamaCppAdapter, Monitor, free_port, preflight
from evaluate_ornith_prompt import MODEL, ROOT, SERVER, require_envelope
from xing_specialist.tools import IsolatedExecutor


CASES = {
    'tenant_reports': {
        'objective': 'A user authorized for two tenants sees the first tenant report after switching to the second. Find and repair the cause.',
        'root': ROOT / 'examples/investigation/tenant_reports',
        'target': 'reporting/service.py',
        'decoy': 'reporting/cache.py',
        'command': ('python3', '-m', 'unittest', 'discover', '-s', 'tests', '-v'),
        'oracle_old': "key = user['id']",
        'oracle_new': "key = (user['id'], tenant)",
    },
    'discount_shipping': {
        'objective': 'A coupon drops an order below the free-shipping threshold, but checkout still waives delivery. Find and repair the cause.',
        'root': ROOT / 'examples/investigation/discount_shipping',
        'target': 'checkout/order.py',
        'decoy': 'checkout/shipping.py',
        'command': ('python3', '-m', 'unittest', 'discover', '-s', 'tests', '-v'),
        'oracle_old': 'delivery = fee(subtotal)',
        'oracle_new': 'delivery = fee(discounted)',
    },
}
SCHEMA = {
    'type': 'object',
    'properties': {
        'action': {'type': 'string', 'enum': ['list', 'read', 'search', 'test', 'final']},
        'target': {'type': 'string'},
        'why': {'type': 'string', 'maxLength': 100},
        'expect': {'type': 'string', 'maxLength': 100},
        'old': {'type': 'string'},
        'new': {'type': 'string'},
    },
    'required': ['action', 'target', 'why', 'expect', 'old', 'new'],
    'additionalProperties': False,
}
SYSTEM = (
    'Investigate the regression using tools. Reply with exactly one JSON object per turn: '
    'action=list|read|search|test|final, target=relative path or search phrase, '
    'why=your current hypothesis in at most 12 words, '
    'expect=predicted observation in at most 12 words, '
    'old and new=exact code fragments for final only (empty otherwise). '
    'Use evidence to reconsider your hypothesis. A final action must propose an exact edit '
    'in target. Do not invent tool results.'
)


def test_result(executor, root, command):
    result = executor.run(root, command)
    return {'exit_code': result.exit_code, 'output': (result.stdout + result.stderr)[-1600:],
            'timed_out': result.timed_out}


def replace_once(root, target, old, new):
    path = root / target
    if (not path.is_file() or not old or not new or old == new
            or max(len(old), len(new)) > 1000
            or max(old.count('\n'), new.count('\n')) > 12):
        return False
    source = path.read_text()
    if source.count(old) != 1:
        return False
    path.write_text(source.replace(old, new, 1))
    return True


def verify_fixture(case, executor):
    baseline = test_result(executor, case['root'], case['command'])
    with tempfile.TemporaryDirectory(prefix='ornith-oracle-') as folder:
        copy = Path(folder) / 'repo'
        shutil.copytree(case['root'], copy)
        if not replace_once(copy, case['target'], case['oracle_old'], case['oracle_new']):
            raise RuntimeError('Oracle edit is not unique')
        repaired = test_result(executor, copy, case['command'])
    if baseline['exit_code'] == 0 or repaired['exit_code'] != 0:
        raise RuntimeError('Fixture needs a failing baseline and passing oracle')
    return {'baseline': baseline, 'oracle': repaired}


def fixture_signature(fixture):
    def stable(result):
        output = re.sub(r'Ran (\d+) tests? in [\d.]+s', r'Ran \1 tests', result['output'])
        return result['exit_code'], result['timed_out'], output
    return stable(fixture['baseline']), stable(fixture['oracle'])


class Investigation:
    def __init__(self, case, executor):
        self.case = case
        self.executor = executor
        self.files = sorted(str(path.relative_to(case['root'])) for path in
                            case['root'].rglob('*.py') if '__pycache__' not in path.parts)
        self.steps = []
        self.inspected = set()

    def observe(self, proposal):
        action = proposal.get('action')
        target = proposal.get('target', '')
        if not isinstance(target, str):
            return 'Invalid target'
        if action == 'list':
            return '\n'.join(self.files)
        if action == 'test':
            return json.dumps(test_result(self.executor, self.case['root'], self.case['command']))[:1700]
        if action == 'read':
            if target not in self.files:
                return 'Unknown path. Use list for allowed files.'
            self.inspected.add(target)
            return (self.case['root'] / target).read_text()[:1700]
        if action == 'search':
            if not target or len(target) > 80:
                return 'Search phrase must contain 1-80 characters.'
            hits = []
            for name in self.files:
                for number, line in enumerate((self.case['root'] / name).read_text().splitlines(), 1):
                    if target.lower() in line.lower():
                        hits.append(f'{name}:{number}: {line[:140]}')
            self.inspected.update(name for name in self.files
                                  if any(hit.startswith(name + ':') for hit in hits))
            return '\n'.join(hits[:12]) or 'No matches'
        return 'Unknown action'

    def final_grade(self, proposal):
        target = proposal.get('target')
        old = proposal.get('old')
        new = proposal.get('new')
        if not all(isinstance(item, str) for item in (target, old, new)):
            return {'passed': False, 'reason': 'Missing edit fields'}
        if target not in self.files or target not in self.inspected:
            return {'passed': False, 'reason': 'Target was not inspected'}
        with tempfile.TemporaryDirectory(prefix='ornith-proposal-') as folder:
            copy = Path(folder) / 'repo'
            shutil.copytree(self.case['root'], copy)
            if not replace_once(copy, target, old, new):
                return {'passed': False, 'reason': 'Edit did not match exactly once'}
            result = test_result(self.executor, copy, self.case['command'])
        return {'passed': target == self.case['target'] and result['exit_code'] == 0,
                'target_correct': target == self.case['target'], 'test': result}

    def metrics(self):
        actions = [step['proposal'].get('action') for step in self.steps]
        targets = [step['proposal'].get('target') for step in self.steps]
        evidence_steps = [step for step in self.steps
                          if step['proposal'].get('action') in ('read', 'search', 'test')]
        decoy = self.case['decoy']
        decoy_read = next((i for i, step in enumerate(self.steps)
                           if step['proposal'].get('action') == 'read'
                           and step['proposal'].get('target') == decoy), None)
        true_read = next((i for i, step in enumerate(self.steps)
                          if step['proposal'].get('action') == 'read'
                          and step['proposal'].get('target') == self.case['target']), None)
        finals = [step for step in self.steps if step['proposal'].get('action') == 'final']
        return {'tool_actions': len(evidence_steps),
                'tested': 'test' in actions,
                'true_file_inspected': self.case['target'] in self.inspected,
                'distinct_files_inspected': len(self.inspected),
                'moved_from_decoy_to_cause': (decoy_read is not None and true_read is not None
                                              and decoy_read < true_read),
                'predictions_recorded': sum(bool(str(step['proposal'].get('expect', '')).strip())
                                            for step in evidence_steps),
                'final_attempts': len(finals),
                'recovered_after_failed_edit': (len(finals) > 1
                                                and any(not step['grade']['passed']
                                                        for step in finals[:-1])
                                                and finals[-1]['grade']['passed']),
                'repeated_action': any((actions[i], targets[i]) == (actions[i - 1], targets[i - 1])
                                       for i in range(1, len(actions))),
                'hypotheses': [step['proposal'].get('why', '') for step in self.steps],
                'predictions': [step['proposal'].get('expect', '') for step in self.steps]}


def parse_answer(raw):
    try:
        value = json.loads(raw.strip())
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def ask(port, prompt):
    payload = json.dumps({'model': 'local', 'messages': [
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
        'temperature': 0, 'max_tokens': 160,
        'response_format': {'type': 'json_schema', 'json_schema': {'schema': SCHEMA}}}).encode()
    start = monotonic()
    with urlopen(Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=payload,
                         headers={'Content-Type': 'application/json'}), timeout=420) as response:
        body = json.load(response)
        answer = body['choices'][0]['message']['content']
    return answer, round(monotonic() - start, 2), body.get('usage', {}), (body.get('timings') or {}).get('predicted_per_second')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def run_case(case, executor, port, max_steps, output, result):
    investigation = Investigation(case, executor)
    investigation.steps = list(result.get('steps', []))
    for step in investigation.steps:
        proposal = step.get('proposal', {})
        if proposal.get('action') == 'read' and proposal.get('target') in investigation.files:
            investigation.inspected.add(proposal['target'])
        if proposal.get('action') == 'search':
            for line in step.get('observation', '').splitlines():
                name = line.split(':', 1)[0]
                if name in investigation.files:
                    investigation.inspected.add(name)
    baseline = result['fixture']['baseline']
    history = ['Initial failing test:\n' + baseline['output'][-700:]]
    for step in investigation.steps:
        proposal = step.get('proposal', {})
        history.append('Action ' + str(step['number']) + ': '
                       + str(proposal.get('action')) + ' '
                       + str(proposal.get('target', ''))[:80]
                       + '\nHypothesis: ' + str(proposal.get('why', ''))[:150]
                       + '\nPrediction: ' + str(proposal.get('expect', ''))[:150]
                       + '\nObserved: ' + step.get('observation', '')[:750])
    for number in range(len(investigation.steps) + 1, max_steps + 1):
        prompt = (case['objective'] + '\nStep ' + str(number) + '/' + str(max_steps)
                  + '. Allowed tools: list, read, search, test; final proposes a fix. '
                  + 'No tool commands are accepted.\n' + '\n'.join(history[-3:])[-1600:])
        raw, seconds, usage, tps = ask(port, prompt)
        proposal = parse_answer(raw)
        action = proposal.get('action')
        if action == 'final':
            grade = investigation.final_grade(proposal)
            feedback = {key: value for key, value in grade.items() if key != 'target_correct'}
            observation = 'Final edit checked against isolated tests: ' + json.dumps(feedback)
        else:
            observation = investigation.observe(proposal)
            grade = None
        step = {'number': number, 'raw': raw[:800], 'proposal': proposal,
                'observation': observation, 'seconds': seconds, 'usage': usage,
                'generation_tps': tps}
        if grade is not None:
            step['grade'] = grade
        investigation.steps.append(step)
        result['steps'] = investigation.steps
        result['metrics'] = investigation.metrics()
        if grade is not None:
            result['grade'] = grade
        save(output, result)
        print(json.dumps({'step': number, 'action': action, 'target': proposal.get('target'),
                          'grade': grade, 'seconds': seconds}), flush=True)
        if action == 'final' and grade['passed']:
            break
        history.append('Action ' + str(number) + ': ' + str(action) + ' '
                       + str(proposal.get('target', ''))[:80]
                       + '\nHypothesis: ' + str(proposal.get('why', ''))[:150]
                       + '\nPrediction: ' + str(proposal.get('expect', ''))[:150]
                       + '\nObserved: ' + observation[:750]
                       + '\nReconsider the hypothesis and choose the next action.')
    return result


def deadline_reached(_signal, _frame):
    raise TimeoutError('Investigation deadline reached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--max-steps', type=int, default=5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true', help='Continue a saved incomplete trace')
    parser.add_argument('--check-envelope', action='store_true')
    parser.add_argument('--model-path', type=Path, default=MODEL)
    parser.add_argument('--model-name', default='Ornith-1.5-9B IQ2_M')
    args = parser.parse_args()
    if not 2 <= args.max_steps <= 8:
        parser.error('--max-steps must be 2-8')
    require_envelope()
    if args.check_envelope:
        print('Ornith CPU and memory envelope is active')
        return 0
    output = args.output or ROOT / ('artifacts/ornith-investigation-' + args.case + '.json')
    case = CASES[args.case]
    with tempfile.TemporaryDirectory(prefix='ornith-investigation-') as folder:
        executor = IsolatedExecutor(Path(folder), timeout_seconds=20, memory_mib=256)
        fixture = verify_fixture(case, executor)
        model = args.model_path.resolve(strict=True)
        preflight(model, 0)
        with model.open('rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        port = free_port()
        engine = LlamaCppAdapter(SERVER, model, 1024, 1, port, 'mmap',
                                 batch_size=128, ubatch_size=64, no_warmup=True,
                                 request_timeout=420, fit='on', numa='isolate')
        if args.model_name == 'Ornith-1.5-9B IQ2_M':
            engine.command += ['--chat-template-kwargs', '{"enable_thinking":false}',
                               '--reasoning', 'off']
        else:
            engine.command += ['--reasoning', 'off']
            if args.model_name.startswith('DeepSeek-R1-'):
                engine.command += ['--reasoning-budget', '0']
        if args.resume:
            result = json.loads(output.read_text())
            if result.get('case') != args.case or result.get('model') != args.model_name:
                raise RuntimeError('Checkpoint does not match case and model')
            if result.get('grade', {}).get('passed') or len(result.get('steps', [])) >= args.max_steps:
                raise RuntimeError('Checkpoint has no remaining steps')
            if fixture_signature(result['fixture']) != fixture_signature(fixture):
                raise RuntimeError('Fixture has changed since checkpoint')
            result.setdefault('interruptions', []).append({
                'status': result.get('status'), 'error': result.get('error'),
                'after_steps': len(result['steps'])})
            result.pop('error', None)
            result['status'] = 'running'
        else:
            result = {'model': args.model_name, 'case': args.case,
                      'fixture': fixture, 'steps': [], 'status': 'running'}
        save(output, result)
        monitor = None
        try:
            engine.start()
            monitor = Monitor(engine.process.pid, 2304 * 1024 ** 2,
                              1536 * 1024 ** 2, 19 * 1024 ** 3,
                              engine.process.terminate, 5.0)
            monitor.start()
            engine.wait_ready(90)
            run_case(case, executor, port, args.max_steps, output, result)
            if monitor.breach_reason:
                raise RuntimeError(monitor.breach_reason)
            result['status'] = 'completed'
        except Exception as exc:
            result['status'] = 'failed'
            result['error'] = monitor.breach_reason if monitor and monitor.breach_reason else str(exc)
        finally:
            if monitor:
                monitor.stop()
                if monitor.samples:
                    result['peak_rss_mib'] = round(max(s['rss_high_water_bytes']
                                                       for s in monitor.samples) / 1024 ** 2, 1)
            engine.stop()
            result['engine_log_tail'] = [line for _, line in engine.lines[-8:]]
            save(output, result)
    print(json.dumps({'case': args.case, 'status': result['status'],
                      'grade': result.get('grade'), 'metrics': result.get('metrics'),
                      'error': result.get('error')}), flush=True)
    return 0 if result['status'] == 'completed' and result.get('grade', {}).get('passed') else 1


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, deadline_reached)
    raise SystemExit(main())
