#!/usr/bin/env python3
"""Capture host and optional process memory counters as JSON."""

import argparse
import datetime
import json
from pathlib import Path
import shutil


def key_values(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            result[key] = value.strip()
    return result


def process_metrics(pid):
    base = Path('/proc') / str(pid)
    status = key_values(base / 'status')
    stat = (base / 'stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    return {
        'pid': pid,
        'name': status.get('Name'),
        'rss_kb': status.get('VmRSS'),
        'virtual_kb': status.get('VmSize'),
        'major_faults': int(fields[9]),
    }


def capture(pids):
    mem = key_values('/proc/meminfo')
    swaps = []
    for line in Path('/proc/swaps').read_text().splitlines()[1:]:
        name, kind, size, used, priority = line.split()
        swaps.append({'name': name, 'type': kind, 'size_kb': int(size),
                      'used_kb': int(used), 'priority': int(priority)})
    disk = shutil.disk_usage('/')
    vmstat = {}
    for line in Path('/proc/vmstat').read_text().splitlines():
        key, value = line.split()
        if key in {'pgmajfault', 'pswpin', 'pswpout', 'oom_kill'}:
            vmstat[key] = int(value)
    data = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'memory_kb': {key: mem.get(key) for key in
                      ('MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'SwapTotal', 'SwapFree')},
        'swap': swaps,
        'swappiness': int(Path('/proc/sys/vm/swappiness').read_text()),
        'root_disk_bytes': {'total': disk.total, 'used': disk.used, 'free': disk.free},
        'vmstat': vmstat,
        'processes': [],
    }
    psi = Path('/proc/pressure/memory')
    data['memory_pressure'] = psi.read_text().strip().splitlines() if psi.exists() else None
    for pid in pids:
        try:
            data['processes'].append(process_metrics(pid))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            data['processes'].append({'pid': pid, 'unavailable': True})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, action='append', default=[], help='Process to measure; repeatable')
    parser.add_argument('--output', type=Path, help='Write JSON to this path')
    args = parser.parse_args()
    data = json.dumps(capture(args.pid), indent=2) + '\n'
    if args.output:
        args.output.write_text(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    main()
