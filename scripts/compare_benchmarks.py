#!/usr/bin/env python3
"""Compare two measured inference artifacts without hiding failed runs."""

import argparse
import json
from pathlib import Path


def sample_peak(result, summary_key, sample_key):
    values = [result.get(summary_key)]
    values.extend(sample.get(sample_key) for sample in result.get('samples', []))
    values = [value for value in values if isinstance(value, (int, float))]
    return max(values) if values else None


def sample_delta(result, summary_key, sample_key):
    samples = result.get('samples', [])
    if len(samples) >= 2 and sample_key in samples[0] and sample_key in samples[-1]:
        return max(0, samples[-1][sample_key] - samples[0][sample_key])
    return result.get(summary_key)


def summarize(result):
    runs = result.get('runs') or []
    peak_rss = sample_peak(result, 'peak_rss_bytes', 'rss_high_water_bytes')
    peak_swap = sample_peak(result, 'peak_system_swap_used_bytes', 'system_swap_used_bytes')
    disk_read = sample_delta(result, 'disk_read_bytes_delta', 'disk_read_bytes')
    return {
        'model': result.get('model_name') or Path(result.get('model', 'unknown')).stem,
        'status': result.get('status', 'unknown'),
        'error': result.get('error'),
        'context_tokens': result.get('context_tokens'),
        'generation_limit_tokens': result.get('generation_limit_tokens'),
        'prompt_sha256': result.get('prompt_sha256'),
        'model_gib': round(result['model_bytes'] / 1024 ** 3, 2) if result.get('model_bytes') else None,
        'startup_seconds': result.get('startup_seconds'),
        'tokens_per_second': runs[0].get('tokens_per_second_after_first_token') if runs else None,
        'time_to_first_token_seconds': runs[0].get('time_to_first_token_seconds') if runs else None,
        'peak_rss_gib': round(peak_rss / 1024 ** 3, 2) if peak_rss is not None else None,
        'peak_swap_gib': round(peak_swap / 1024 ** 3, 2) if peak_swap is not None else None,
        'major_faults': sample_delta(result, 'major_faults_delta', 'major_faults'),
        'disk_read_gib': round(disk_read / 1024 ** 3, 2) if disk_read is not None else None,
    }


def compare(left, right):
    a, b = summarize(left), summarize(right)
    issues = []
    if a['status'] != 'passed' or b['status'] != 'passed':
        issues.append('Both models must complete inference before performance can be compared')
    for key in ('context_tokens', 'generation_limit_tokens', 'prompt_sha256'):
        if a[key] is None or b[key] is None or a[key] != b[key]:
            issues.append(f'{key} must match and be recorded for both runs')
    return {'left': a, 'right': b, 'comparable': not issues, 'issues': issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    print(json.dumps(compare(left, right), indent=2))


if __name__ == '__main__':
    main()
