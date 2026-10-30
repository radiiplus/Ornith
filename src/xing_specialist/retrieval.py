"""CPU-only FTS5 retrieval over verified engineering lessons."""

import re

from .memory import MemoryStore


class LessonRetriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def search(self, query: str, limit: int = 5) -> list:
        if not 1 <= limit <= 20:
            raise ValueError('Retrieval limit must be between 1 and 20')
        terms = re.findall(r'[A-Za-z0-9_]+', query.lower())[:8]
        if not terms:
            return []
        expression = ' OR '.join('"' + term + '"' for term in terms)
        rows = self.store.db.execute('''SELECT l.id,l.body,l.evidence_test_id,bm25(lesson_fts) AS rank
            FROM lesson_fts JOIN lessons AS l ON l.id=lesson_fts.rowid
            WHERE lesson_fts MATCH ? ORDER BY rank LIMIT ?''', (expression, limit)).fetchall()
        return [dict(row) for row in rows]
