#!/usr/bin/env python3
"""Probe two simultaneous local model requests without changing model state."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from time import monotonic
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8082/v1/chat/completions')
    parser.add_argument('--model', default='local')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()

    def request(index):
        payload = json.dumps({'model': args.model, 'messages': [
            {'role': 'user', 'content': f'Return only the number {index}.'}],
            'max_tokens': 24, 'temperature': 0, 'stream': False}).encode()
        started = monotonic()
        with urlopen(Request(args.url, data=payload,
                             headers={'Content-Type': 'application/json'}),
                     timeout=args.timeout) as response:
            result = json.load(response)
        return {'request': index, 'seconds': round(monotonic() - started, 2),
                'answer': result['choices'][0]['message']['content'][:100]}

    started = monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, (1, 2)))
    print(json.dumps({'wall_seconds': round(monotonic() - started, 2), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
