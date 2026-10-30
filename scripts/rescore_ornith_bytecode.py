#!/usr/bin/env python3
"""Replay saved Ornith bytecode answers through the current deterministic scorer."""

import argparse
import json
from pathlib import Path

from evaluate_ornith_bytecode import assess_answer, assess_reasoned_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if not data['ground_truth']['verified']:
        raise ValueError('Fixture ground truth is not verified')
    for row in data['responses']:
        scorer = (assess_reasoned_answer(row['raw_answer'], data['ground_truth'])
                  if row['mode'] in ('evidence_first', 'feedback_revision')
                  else assess_answer(row['raw_answer']))
        row.update(scorer)
    data['rescored_from'] = str(args.input)
    output = args.output or args.input.with_name(args.input.stem + '-rescored.json')
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'grades': {
        row['mode']: row['passed'] for row in data['responses']}}, indent=2))


if __name__ == '__main__':
    main()
