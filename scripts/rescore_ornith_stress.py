#!/usr/bin/env python3
"""Replay saved Ornith responses through the current objective verifier."""

import argparse
import json
from pathlib import Path

from stress_ornith_reasoning import CHALLENGES, assess_response
from prepare_verified_repairs import load_cases


def rescore(source):
    saved = json.loads(source.read_text())
    cases = {row['id']: row for row in load_cases(CHALLENGES)}
    results = []
    for item in saved['cases']:
        row = cases[item['id']]
        attempts = []
        for original in item['attempts']:
            assessment = assess_response(row, original['content'], saved.get('tool_assisted', False))
            attempts.append({'attempt': original['attempt'], **assessment,
                             'seconds': original['seconds'], 'usage': original['usage']})
            if assessment['passed']:
                break
        results.append({'id': item['id'], 'domain': item['domain'],
                        'attempts': attempts, 'passed': bool(attempts and attempts[-1]['passed'])})
    return {'source': str(source), 'model': saved['model'], 'thinking': saved['thinking'],
            'cases': results, 'objectives_reached': sum(item['passed'] for item in results),
            'attempts_after_first_success_ignored': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = rescore(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'objectives_reached': result['objectives_reached'],
                      'cases': len(result['cases']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
