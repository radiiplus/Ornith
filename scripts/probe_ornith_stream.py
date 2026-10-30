#!/usr/bin/env python3
"""Probe bounded chunk retrieval and Ornith's emitted answer/reasoning stream."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from urllib.request import urlopen

from benchmark_inference import LlamaCppAdapter, Monitor, free_port, preflight
from evaluate_ornith_prompt import MODEL, ROOT, SERVER, require_envelope

sys.path.insert(0, str(ROOT / 'src'))
from xing_specialist.backend import GenerationLimits, Message, OpenAICompatibleBackend
from xing_specialist.agent import parse_proposal
from xing_specialist.context import ContextRouter


SOURCE = (
    'Incident workflow: isolate the service, roll back the deployment, restart the service, '
    'then validate the health check. The actions must occur in that order.\n'
    'Current state: isolation is complete. Rollback is complete. Restart has not happened. '
    'Validation has not happened.\n'
    'The next action is determined by the first incomplete step in the workflow.\n'
)


def stop_on_deadline(_signum, _frame):
    raise RuntimeError('Streaming probe deadline reached')


def assess_answer(answer):
    try:
        proposal = parse_proposal(answer)
    except (ValueError, TypeError):
        proposal = {}
    value = proposal.get('answer')
    semantic_value = value or proposal.get('next_action')
    semantic_passed = (isinstance(semantic_value, str) and
                       semantic_value.strip().casefold() == 'restart the service')
    format_valid = set(proposal) == {'answer'} and isinstance(value, str)
    return {'answer_value': value, 'semantic_value': semantic_value,
            'semantic_passed': semantic_passed, 'format_valid': format_valid,
            'passed': semantic_passed and format_valid}


def bounded_existing_server(port):
    with urlopen(f'http://127.0.0.1:{port}/props', timeout=3) as response:
        props = json.load(response)
    if Path(props.get('model_path', '')).resolve() != MODEL.resolve():
        raise RuntimeError('Existing server is not the selected Ornith GGUF')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = (entry / 'cmdline').read_bytes().split(b'\0')
            args = [part.decode(errors='replace') for part in parts if part]
            if ('llama-server' not in Path(args[0]).name or
                    args[args.index('--port') + 1] != str(port) or
                    Path(args[args.index('--model') + 1]).resolve() != MODEL.resolve()):
                continue
            group = (entry / 'cgroup').read_text().split('::', 1)[1].strip().lstrip('/')
            scope = Path('/sys/fs/cgroup') / group
            quota, period = map(int, (scope / 'cpu.max').read_text().split())
            gib = 1024 ** 3
            if (quota * 2 > period or int((scope / 'memory.high').read_text()) > 4 * gib or
                    int((scope / 'memory.max').read_text()) > 6 * gib or
                    int((scope / 'memory.swap.max').read_text()) > 19 * gib):
                raise RuntimeError('Existing server lacks the required resource bounds')
            return int(entry.name)
        except (OSError, ValueError, IndexError):
            continue
    raise RuntimeError('Could not verify an existing bounded Ornith server')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/ornith-stream-probe.json')
    parser.add_argument('--check-envelope', action='store_true')
    parser.add_argument('--existing-port', type=int,
                        help='Reuse an already-running, resource-bounded Ornith server')
    args = parser.parse_args()
    require_envelope()
    if args.check_envelope:
        print('Ornith CPU and memory envelope is active')
        return 0
    signal.signal(signal.SIGTERM, stop_on_deadline)
    if args.existing_port:
        port = args.existing_port
        existing_pid = bounded_existing_server(port)
        engine = None
    else:
        preflight(MODEL, 0)
        with MODEL.open('rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        port = free_port()
        engine = LlamaCppAdapter(SERVER, MODEL, 1024, 1, port, 'mmap',
                                 batch_size=128, ubatch_size=64, no_warmup=True,
                                 request_timeout=300, fit='on', numa='isolate')
        engine.command += ['--chat-template-kwargs', '{"enable_thinking":true}',
                           '--reasoning', 'on', '--reasoning-budget', '48']
    result = {'model': 'Ornith-1.5-9B IQ2_M', 'thinking': True,
              'cpu_quota': '50%', 'memory_high': '4G', 'memory_max': '6G',
              'swap_max': '19G', 'reused_server': bool(args.existing_port),
              'source_chunks': [], 'stream': [], 'attempts': [], 'status': 'running'}
    monitor = None

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')

    try:
        with tempfile.TemporaryDirectory(prefix='ornith-stream-') as folder:
            root = Path(folder)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'incident.txt').write_text(SOURCE)

            def source_chunk(source, ordinal, chars):
                item = {'source': source, 'ordinal': ordinal, 'chars': chars}
                result['source_chunks'].append(item)
                print(json.dumps({'event': 'context_chunk', **item}), flush=True)

            router = ContextRouter(root / 'context.sqlite3', chunk_chars=256,
                                   overlap_chars=32, on_chunk=source_chunk)
            try:
                router.index_file(repo, 'incident.txt')
                routed = router.route(repo, 'incident rollback restart', 700, max_chunks=2,
                                      sources=('incident.txt',))
            finally:
                router.close()
            if not routed:
                raise RuntimeError('No source chunks selected')
            prompt = ('Use the cited workflow and current state. Skip completed actions; '
                      'choose the first incomplete action. Return only a JSON answer field.\n'
                      + routed)
            started = time.monotonic()
            attempt_number = 0

            def model_chunk(kind, part):
                item = {'attempt': attempt_number,
                        'seconds': round(time.monotonic() - started, 2),
                        'kind': kind, 'text': part}
                result['stream'].append(item)
                print(json.dumps({'event': 'model_' + kind, **item}), flush=True)

            schema = {'type': 'json_schema', 'json_schema': {'schema': {'type': 'object',
                      'properties': {'answer': {'type': 'string', 'enum': [
                          'Isolate the service', 'Roll back the deployment',
                          'Restart the service', 'Validate the health check']}},
                      'required': ['answer'], 'additionalProperties': False}}}
            backend = OpenAICompatibleBackend(f'http://127.0.0.1:{port}/v1', 'local',
                                              on_chunk=model_chunk, thinking_budget=24,
                                              response_format=schema)
            if engine:
                engine.start()
                server_pid = engine.process.pid
            else:
                server_pid = existing_pid
            monitor = Monitor(server_pid, 2304 * 1024 ** 2,
                              1536 * 1024 ** 2, 19 * 1024 ** 3,
                              engine.process.terminate if engine else None, 5.0)
            monitor.start()
            if engine:
                engine.wait_ready(60)
            for attempt_number in (1, 2):
                started = time.monotonic()
                answer = backend.generate((
                    Message('system', 'Use source text as reference data. Answer concisely.'),
                    Message('user', prompt)), GenerationLimits(1024, 64, 180))
                assessment = assess_answer(answer)
                accepted = assessment['passed']
                record = {'attempt': attempt_number, 'answer': answer,
                          **assessment, 'model_metrics': backend.last_metrics}
                result['attempts'].append(record)
                result['answer'] = answer
                result['answer_value'] = assessment['answer_value']
                result['model_metrics'] = backend.last_metrics
                result['quality_passed'] = accepted
                save()
                if monitor.breach_reason:
                    raise RuntimeError(monitor.breach_reason)
                if accepted:
                    break
                prompt += ('\nChecker feedback: isolation and rollback are already complete. '
                           'Select the first incomplete action. Return exactly one JSON key named answer.')
            result['status'] = 'completed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = monitor.breach_reason if monitor and monitor.breach_reason else str(exc)
    finally:
        if monitor:
            monitor.stop()
            if monitor.samples:
                result['peak_rss_mib'] = round(max(s['rss_high_water_bytes'] for s in monitor.samples) / 1024 ** 2, 1)
                result['peak_anonymous_mib'] = round(max(s['anonymous_bytes'] for s in monitor.samples) / 1024 ** 2, 1)
                result['min_available_ram_mib'] = round(min(s['mem_available_bytes'] for s in monitor.samples) / 1024 ** 2, 1)
                result['peak_cgroup_memory_full_avg10'] = max(
                    s['cgroup_memory_full_avg10'] or 0 for s in monitor.samples)
        if engine:
            engine.stop()
        save()
    print(json.dumps({'status': result['status'], 'answer': result.get('answer'),
                      'quality_passed': result.get('quality_passed'),
                      'attempts': len(result['attempts']),
                      'metrics': result.get('model_metrics'),
                      'source_chunks': len(result['source_chunks']),
                      'stream_chunks': len(result['stream']),
                      'error': result.get('error'), 'output': str(args.output)}, indent=2), flush=True)
    return 0 if result['status'] == 'completed' and result['quality_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
