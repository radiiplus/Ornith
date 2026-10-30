#!/usr/bin/env python3
"""Rescore saved objective restatements with the current deterministic gate."""

import argparse
import json
from pathlib import Path

from evaluate_candidate_roles import GATE_IDS, REPAIR_CASES
from prepare_verified_repairs import load_cases
from stress_ornith_reasoning import CHALLENGES, SEED
from xing_specialist.dual_router import objective_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    args = parser.parse_args()
    result = json.loads(args.artifact.read_text())
    cases = {row['id']: row['question'] for row in load_cases(CHALLENGES) +
             load_cases(SEED) if row.get('verifier') and row['id'] in GATE_IDS}
    cases['tenant_reports'] = REPAIR_CASES['tenant_reports']['objective']
    for record in result['objective_gates']:
        try:
            objective_gate(cases[record['case']], record['response'])
            record['objective_fidelity'] = True
            record.pop('feedback', None)
        except (ValueError, TypeError, KeyError) as exc:
            record['objective_fidelity'] = False
            record['feedback'] = str(exc)[:240]
    args.artifact.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'artifact': str(args.artifact), 'objective_fidelity':
                      sum(row['objective_fidelity'] for row in result['objective_gates']),
                      'cases': len(result['objective_gates'])}))


if __name__ == '__main__':
    main()
