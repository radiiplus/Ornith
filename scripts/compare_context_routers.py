#!/usr/bin/env python3
"""Compare the SQLite router and Rust stream scanner on one synthetic corpus."""

import json
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from xing_specialist.context import ContextRouter


def elapsed(call):
    started = time.perf_counter()
    result = call()
    return time.perf_counter() - started, result


def main():
    binary = ROOT / 'experiments/rust-context-probe/target/release/rust-context-probe'
    if not binary.is_file():
        raise SystemExit('Build the Rust release binary with cargo build --release --manifest-path experiments/rust-context-probe/Cargo.toml')

    with tempfile.TemporaryDirectory(prefix='xing-router-compare-') as folder:
        base = Path(folder)
        repo = base / 'repo'
        repo.mkdir()
        source = repo / 'corpus.txt'
        with source.open('w') as stream:
            for _ in range(100):
                stream.write('alpha beta gamma delta epsilon ' * 2000)
            stream.write(' parser_deadline_overflow requires signed bounds validation')

        router = ContextRouter(base / 'context.sqlite3')
        try:
            index_seconds, chunks = elapsed(lambda: router.index_file(repo, 'corpus.txt'))
            sqlite_times = []
            rust_times = []
            for _ in range(10):
                duration, routed = elapsed(lambda: router.route(repo, 'parser_deadline_overflow', 1200))
                sqlite_times.append(duration)
                if 'parser_deadline_overflow' not in routed:
                    raise RuntimeError('SQLite missed the marker')

                duration, run = elapsed(lambda: subprocess.run(
                    [str(binary), str(source), 'parser_deadline_overflow', '1200'],
                    capture_output=True, text=True, check=True, timeout=10))
                rust_times.append(duration)
                if 'parser_deadline_overflow' not in run.stdout:
                    raise RuntimeError('Rust missed the marker')

            print(json.dumps({
                'source_bytes': source.stat().st_size,
                'sqlite_chunks': chunks,
                'sqlite_index_seconds': round(index_seconds, 4),
                'sqlite_query_median_ms': round(statistics.median(sqlite_times) * 1000, 3),
                'rust_query_median_ms_with_process_start': round(statistics.median(rust_times) * 1000, 3),
                'python_peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                'rust_peak_rss_mib': round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
            }, indent=2))
        finally:
            router.close()


if __name__ == '__main__':
    main()
