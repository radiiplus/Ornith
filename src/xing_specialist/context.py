"""Disk-backed, bounded retrieval over user-selected source files."""

from pathlib import Path
from hashlib import sha256
import re
import sqlite3

from .memory import redact


class ContextRouter:
    def __init__(self, path: Path, chunk_chars: int = 800, overlap_chars: int = 100,
                 on_chunk=None):
        if chunk_chars < 256 or not 0 <= overlap_chars < chunk_chars:
            raise ValueError('Invalid context chunk dimensions')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode = WAL')
        self.db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS context_fts USING fts5(scope UNINDEXED, source, ordinal UNINDEXED, body)')
        self.db.execute('''CREATE TABLE IF NOT EXISTS context_index_state (
            scope TEXT NOT NULL, source TEXT NOT NULL, digest TEXT NOT NULL,
            chunk_count INTEGER NOT NULL, chunk_chars INTEGER NOT NULL,
            overlap_chars INTEGER NOT NULL, PRIMARY KEY(scope,source))''')
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.last_index_reused = False
        self.on_chunk = on_chunk

    def close(self):
        self.db.close()

    def index_file(self, repository: Path, name: str) -> int:
        root = Path(repository).resolve(strict=True)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or name.startswith('.'):
            raise ValueError(f'Unsafe context source: {name}')
        target = root / relative
        if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(root):
            raise ValueError(f'Invalid context source: {name}')
        if target.stat().st_size > 128 * 1024 * 1024:
            raise ValueError(f'Context source exceeds 128 MiB: {name}')

        before = target.stat()
        digest = sha256()
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        after = target.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError(f'Context source changed while hashing: {name}')
        state = self.db.execute('''SELECT digest,chunk_count,chunk_chars,overlap_chars
            FROM context_index_state WHERE scope=? AND source=?''', (str(root), name)).fetchone()
        indexed = self.db.execute('SELECT COUNT(*) FROM context_fts WHERE scope=? AND source=?',
                                  (str(root), name)).fetchone()[0]
        if (state and state['digest'] == digest.hexdigest() and state['chunk_chars'] == self.chunk_chars
                and state['overlap_chars'] == self.overlap_chars and indexed == state['chunk_count']):
            self.last_index_reused = True
            return state['chunk_count']

        self.last_index_reused = False
        count = 0
        with target.open('r', encoding='utf-8', errors='replace') as stream, self.db:
            self.db.execute('DELETE FROM context_fts WHERE scope=? AND source=?', (str(root), name))
            chunk = stream.read(self.chunk_chars)
            while chunk:
                self.db.execute('INSERT INTO context_fts(scope, source, ordinal, body) VALUES (?,?,?,?)',
                                (str(root), name, count, redact(chunk)))
                count += 1
                next_text = stream.read(self.chunk_chars - self.overlap_chars)
                chunk = chunk[-self.overlap_chars:] + next_text if next_text and self.overlap_chars else next_text
            final = target.stat()
            if (after.st_size, after.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                raise OSError(f'Context source changed while indexing: {name}')
            self.db.execute('''INSERT INTO context_index_state
                (scope,source,digest,chunk_count,chunk_chars,overlap_chars) VALUES (?,?,?,?,?,?)
                ON CONFLICT(scope,source) DO UPDATE SET digest=excluded.digest,
                chunk_count=excluded.chunk_count, chunk_chars=excluded.chunk_chars,
                overlap_chars=excluded.overlap_chars''',
                (str(root), name, digest.hexdigest(), count, self.chunk_chars, self.overlap_chars))
        return count

    def route(self, repository: Path, query: str, max_chars: int, max_chunks: int = 6,
              sources: tuple = ()) -> str:
        if max_chars < 0 or not 1 <= max_chunks <= 20:
            raise ValueError('Invalid context routing budget')
        terms = re.findall(r'[A-Za-z0-9_]+', query.lower())[:12]
        if not terms or max_chars == 0:
            return ''
        expression = ' OR '.join('"' + term + '"' for term in terms)
        source_filter = ' AND source IN (' + ','.join('?' for _ in sources) + ')' if sources else ''
        rows = self.db.execute('''SELECT source, ordinal, body FROM context_fts
            WHERE context_fts MATCH ? AND scope=?''' + source_filter +
            ' ORDER BY bm25(context_fts) LIMIT ?',
            (expression, str(Path(repository).resolve(strict=True)), *sources, max_chunks * 4)).fetchall()
        selected = []
        remaining = max_chars
        for row in rows:
            section = f'[SOURCE {row["source"]} chunk {row["ordinal"]}]\n{row["body"]}'
            if len(section) > remaining:
                continue
            selected.append(section)
            if self.on_chunk:
                self.on_chunk(row['source'], row['ordinal'], len(row['body']))
            remaining -= len(section) + 2
            if len(selected) == max_chunks or remaining <= 0:
                break
        return '\n\n'.join(selected)
