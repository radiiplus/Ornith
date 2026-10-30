#!/usr/bin/env python3
"""Measure disk-backed routing over a synthetic million-token corpus."""

import json
from pathlib import Path
import resource
import sys
import tempfile
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from xing_specialist.context import ContextRouter


def main():
    with tempfile.TemporaryDirectory(prefix='xing-context-probe-') as folder:
        root = Path(folder)
        repo = root / 'repo'
        repo.mkdir()
        source = repo / 'corpus.txt'
        with source.open('w') as stream:
            for _ in range(100):
                stream.write('alpha beta gamma delta epsilon ' * 2000)
            stream.write(' parser_deadline_overflow requires signed bounds validation')
        router = ContextRouter(root / 'context.sqlite3')
        started = time.monotonic()
        try:
            chunks = router.index_file(repo, 'corpus.txt')
            indexed_seconds = time.monotonic() - started
            routed = router.route(repo, 'parser_deadline_overflow', 1200)
            result = {
                'synthetic_tokens': 1_000_000,
                'source_bytes': source.stat().st_size,
                'index_bytes': (root / 'context.sqlite3').stat().st_size,
                'chunks': chunks,
                'index_seconds': round(indexed_seconds, 3),
                'retrieved_marker': 'parser_deadline_overflow' in routed,
                'retrieved_chars': len(routed),
                'process_peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            }
            print(json.dumps(result, indent=2))
            if not result['retrieved_marker']:
                raise SystemExit(1)
        finally:
            router.close()


if __name__ == '__main__':
    main()
