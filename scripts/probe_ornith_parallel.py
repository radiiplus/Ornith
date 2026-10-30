#!/usr/bin/env python3
"""Run two short Ornith requests under the established resource envelope."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
from time import monotonic
from urllib.request import Request, urlopen

from benchmark_inference import LlamaCppAdapter, Monitor, free_port, preflight
from evaluate_ornith_prompt import MODEL, ROOT, SERVER, require_envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/ornith-parallel-probe.json')
    args = parser.parse_args()
    require_envelope()
    preflight(MODEL, 0)
    with MODEL.open('rb') as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    port = free_port()
    engine = LlamaCppAdapter(SERVER, MODEL, 2048, 1, port, 'mmap',
                             batch_size=128, ubatch_size=64, no_warmup=True,
                             request_timeout=180, fit='on', numa='isolate', parallel=2)
    engine.command += ['--chat-template-kwargs', '{"enable_thinking":false}',
                       '--reasoning', 'off']
    monitor = None
    try:
        engine.start()
        monitor = Monitor(engine.process.pid, 2304 * 1024 ** 2,
                          1536 * 1024 ** 2, 19 * 1024 ** 3,
                          engine.process.terminate, 5.0)
        monitor.start()
        engine.wait_ready(90)

        def ask(index):
            body = json.dumps({'model': 'local', 'messages': [
                {'role': 'user', 'content': f'Reply with only {index}.'}],
                'max_tokens': 12, 'temperature': 0, 'stream': False}).encode()
            started = monotonic()
            with urlopen(Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 data=body, headers={'Content-Type': 'application/json'}),
                         timeout=180) as response:
                answer = json.load(response)['choices'][0]['message']['content']
            return {'request': index, 'seconds': round(monotonic() - started, 2),
                    'answer': answer[:100]}

        started = monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(ask, (1, 2)))
        output = {'model': 'Ornith-1.5-9B IQ2_M', 'parallel': 2,
                  'context_per_slot': 1024, 'wall_seconds': round(monotonic() - started, 2),
                  'answers': answers, 'memory_breach': monitor.breach_reason,
                  'peak_rss_mib': round(max(s['rss_high_water_bytes'] for s in monitor.samples)
                                        / 1024 ** 2, 1) if monitor.samples else None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + '\n')
        print(json.dumps(output, indent=2), flush=True)
        return 0 if not monitor.breach_reason and all(
            answer['answer'].strip() == str(answer['request']) for answer in answers) else 1
    finally:
        if monitor:
            monitor.stop()
        engine.stop()


if __name__ == '__main__':
    raise SystemExit(main())
