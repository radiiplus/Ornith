#!/usr/bin/env python3
"""Rescore saved stream responses without loading a model."""

import argparse
import json
from pathlib import Path

from probe_ornith_stream import assess_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = json.loads(args.input.read_text())
    attempts = result.get('attempts', [])
    for attempt in attempts:
        attempt.update(assess_answer(attempt['answer']))
    if attempts:
        result['answer'] = attempts[-1]['answer']
        result['answer_value'] = attempts[-1]['answer_value']
        result['quality_passed'] = any(attempt['passed'] for attempt in attempts)
    result['scoring_revision'] = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'quality_passed': result.get('quality_passed'),
                      'attempts': [{'attempt': row['attempt'], 'passed': row['passed'],
                                    'semantic_passed': row['semantic_passed'],
                                    'format_valid': row['format_valid']}
                                   for row in attempts],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
