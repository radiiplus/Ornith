"""Collect source-backed references for general and workspace research."""

import argparse
import json
from pathlib import Path

from .backend import BackendError, GenerationLimits, OpenAICompatibleBackend
from .research import ResearchLibrary, browse, extract_findings, local_source


def main():
    parser = argparse.ArgumentParser(description='Index bounded local and web research')
    parser.add_argument('--objective', required=True)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--local-source', action='append', default=[])
    parser.add_argument('--seed-url', action='append', default=[])
    parser.add_argument('--max-pages', type=int, default=4)
    parser.add_argument('--extract', action='store_true',
                        help='Ask the local model for quote-backed candidate claims and procedures')
    parser.add_argument('--backend-url', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--model-id', default='local')
    parser.add_argument('--context', type=int, default=1024)
    parser.add_argument('--max-output', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--state-dir', type=Path,
                        default=Path.home() / '.local' / 'share' / 'xing-specialist')
    args = parser.parse_args()
    if args.local_source and (not args.workspace or len(args.local_source) > 20):
        parser.error('Local sources require a workspace and are limited to 20 files')
    library = ResearchLibrary(args.state_dir / 'research.sqlite3')
    try:
        sources = [local_source(args.workspace, name) for name in args.local_source]
        if args.seed_url:
            sources.extend(browse(tuple(args.seed_url), args.objective, args.max_pages))
        backend = (OpenAICompatibleBackend(args.backend_url, args.model_id) if args.extract else None)
        limits = GenerationLimits(args.context, args.max_output, args.timeout)
        result = []
        for source in sources:
            source_id = library.add_source(source)
            findings = []
            extraction_error = None
            if backend:
                try:
                    findings = extract_findings(backend, limits, library, source_id,
                                                args.objective)
                except (BackendError, ValueError, TypeError, KeyError) as exc:
                    extraction_error = str(exc)
            result.append({'id': source_id, 'scope': source.scope, 'locator': source.locator,
                           'finding_ids': findings, 'extraction_error': extraction_error})
        print(json.dumps({'sources': result,
                          'matches': library.search(args.objective, args.workspace),
                          'candidate_findings': library.search_findings(args.objective, args.workspace)},
                         indent=2))
        return 0
    finally:
        library.close()


if __name__ == '__main__':
    raise SystemExit(main())
