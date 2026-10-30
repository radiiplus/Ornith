"""Small, evidence-gated memory for recurring gaps and verified procedures."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
from typing import Optional

from .memory import MemoryStore, redact


EXCLUDED_DIRS = {'.git', '.aws', '.env', '.venv', 'node_modules', 'vendor', 'models',
                 'artifacts', '__pycache__', '.pytest_cache'}
STOPWORDS = {'a', 'and', 'code', 'fix', 'for', 'in', 'is', 'of', 'on', 'the', 'to',
             'with', 'test', 'tests', 'problem', 'solve'}


def repository_snapshot_key(repository: Path, objective: str, allowed_paths: tuple,
                            commands: tuple, context_sources: tuple) -> Optional[str]:
    """Return a content key only for repositories within the executor's copy budget."""
    root = Path(repository).resolve(strict=True)
    digest = sha256(json.dumps([str(root), objective, allowed_paths, commands, context_sources],
                               sort_keys=True).encode())
    total = count = 0
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
        for name in sorted(files):
            path = Path(base) / name
            if path.is_symlink():
                return None
            if name.endswith(('.pem', '.key', '.sqlite3', '.gguf')) or name.startswith('.env'):
                continue
            if not path.is_file():
                return None
            before = path.stat()
            size = before.st_size
            total += size
            count += 1
            if total > 50 * 1024 ** 2 or count > 3000:
                return None
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b'\0')
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns):
                return None
    return digest.hexdigest()


class AdaptiveMemory:
    def __init__(self, store: MemoryStore, max_items: int = 128):
        if max_items < 1:
            raise ValueError('Memory limit must be positive')
        self.store = store
        self.max_items = max_items
        self.db = store.db
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS attention_memory (
                id INTEGER PRIMARY KEY, scope TEXT NOT NULL, key TEXT NOT NULL,
                kind TEXT NOT NULL, topic TEXT NOT NULL, body TEXT NOT NULL,
                status TEXT NOT NULL, salience INTEGER NOT NULL,
                seen INTEGER NOT NULL, successes INTEGER NOT NULL,
                failures INTEGER NOT NULL, version INTEGER NOT NULL,
                evidence_test_id INTEGER REFERENCES test_results(id),
                updated_at REAL NOT NULL, UNIQUE(scope,key));
            CREATE INDEX IF NOT EXISTS attention_scope_status
                ON attention_memory(scope,status);
            CREATE TABLE IF NOT EXISTS verified_proposals (
                key TEXT PRIMARY KEY, scope TEXT NOT NULL, answer TEXT NOT NULL,
                skill_id INTEGER REFERENCES attention_memory(id) ON DELETE SET NULL,
                evidence_test_id INTEGER NOT NULL REFERENCES test_results(id),
                active INTEGER NOT NULL, uses INTEGER NOT NULL,
                updated_at REAL NOT NULL);
        ''')

    @staticmethod
    def _key(kind: str, topic: str) -> str:
        normalized = ' '.join(topic.lower().split())
        return kind + ':' + sha256(normalized.encode()).hexdigest()

    def _prune(self):
        rows = self.db.execute('''SELECT id FROM attention_memory
            ORDER BY (status='active') DESC, salience DESC, successes DESC,
                     updated_at DESC LIMIT -1 OFFSET ?''', (self.max_items,)).fetchall()
        if rows:
            self.db.executemany('DELETE FROM attention_memory WHERE id=?',
                                ((row['id'],) for row in rows))

    def observe_gap(self, scope: str, topic: str, category: str, detail: str) -> int:
        if not category or not topic.strip():
            raise ValueError('A gap needs a topic and category')
        key = self._key('gap:' + category, topic)
        body = redact(f'Open question for {topic[:120]}: {detail[:280]}')[:400]
        stamp = time.time()
        with self.db:
            self.db.execute('''INSERT INTO attention_memory
                (scope,key,kind,topic,body,status,salience,seen,successes,failures,version,updated_at)
                VALUES (?,?,?,?,?,'pending',2,1,0,0,1,?)
                ON CONFLICT(scope,key) DO UPDATE SET
                    body=excluded.body, seen=seen+1,
                    salience=MIN(8,salience+1),
                    status=CASE WHEN seen+1>=2 THEN 'open' ELSE 'pending' END,
                    updated_at=excluded.updated_at''',
                (scope, key, 'gap', redact(topic)[:200], body, stamp))
            self._prune()
        return self.db.execute('SELECT id FROM attention_memory WHERE scope=? AND key=?',
                               (scope, key)).fetchone()['id']

    def resolve_gaps(self, scope: str, topic: str):
        with self.db:
            self.db.execute('''UPDATE attention_memory SET status='retired', updated_at=?
                WHERE scope=? AND kind='gap' AND topic=? AND status IN ('pending','open')''',
                (time.time(), scope, redact(topic)[:200]))

    def remember_skill(self, scope: str, topic: str, body: str, evidence_test_id: int) -> int:
        row = self.db.execute('SELECT passed FROM test_results WHERE id=?',
                              (evidence_test_id,)).fetchone()
        if row is None or not row['passed']:
            raise ValueError('A procedure needs passing test evidence')
        if not topic.strip() or not body.strip():
            raise ValueError('A procedure needs a topic and body')
        key = self._key('skill', topic)
        clean_body = redact(body)[:500]
        stamp = time.time()
        with self.db:
            self.db.execute('''INSERT INTO attention_memory
                (scope,key,kind,topic,body,status,salience,seen,successes,failures,version,
                 evidence_test_id,updated_at)
                VALUES (?,?,?,?,?,'active',6,1,1,0,1,?,?)
                ON CONFLICT(scope,key) DO UPDATE SET
                    body=excluded.body, status='active', salience=MIN(10,salience+1),
                    seen=seen+1, successes=successes+1,
                    failures=CASE WHEN body!=excluded.body THEN 0 ELSE failures END,
                    version=version+CASE WHEN body!=excluded.body THEN 1 ELSE 0 END,
                    evidence_test_id=excluded.evidence_test_id, updated_at=excluded.updated_at''',
                (scope, key, 'skill', redact(topic)[:200], clean_body, evidence_test_id, stamp))
            self._prune()
        return self.db.execute('SELECT id FROM attention_memory WHERE scope=? AND key=?',
                               (scope, key)).fetchone()['id']

    def recall(self, scope: str, query: str, limit: int = 3) -> list:
        if not 1 <= limit <= 5:
            raise ValueError('Recall limit must be between one and five')
        terms = set(re.findall(r'[a-z0-9_]+', query.lower())) - STOPWORDS
        if not terms:
            return []
        rows = self.db.execute('''SELECT * FROM attention_memory
            WHERE scope=? AND status IN ('active','open')''', (scope,)).fetchall()
        scored = []
        for row in rows:
            text_terms = (set(re.findall(r'[a-z0-9_]+',
                                         (row['topic'] + ' ' + row['body']).lower())) - STOPWORDS)
            overlap = len(terms & text_terms)
            if overlap:
                score = overlap * 10 + row['salience'] + min(row['successes'], 3) - 2 * row['failures']
                scored.append((score, row['updated_at'], row))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [{'id': row['id'], 'kind': row['kind'], 'body': row['body'],
                 'version': row['version']} for _, _, row in scored[:limit]]

    def store_proposal(self, key: Optional[str], scope: str, answer: str,
                       evidence_test_id: int, skill_id: int):
        if not key or len(answer) > 16384 or redact(answer) != answer:
            return False
        row = self.db.execute('SELECT passed FROM test_results WHERE id=?',
                              (evidence_test_id,)).fetchone()
        if row is None or not row['passed']:
            raise ValueError('A cached proposal needs passing test evidence')
        with self.db:
            self.db.execute('''INSERT INTO verified_proposals
                (key,scope,answer,skill_id,evidence_test_id,active,uses,updated_at)
                VALUES (?,?,?,?,?,1,0,?)
                ON CONFLICT(key) DO UPDATE SET answer=excluded.answer, skill_id=excluded.skill_id,
                    evidence_test_id=excluded.evidence_test_id, active=1,
                    updated_at=excluded.updated_at''',
                (key, scope, answer, skill_id, evidence_test_id, time.time()))
            self.db.execute('''DELETE FROM verified_proposals WHERE key IN
                (SELECT key FROM verified_proposals ORDER BY active DESC, updated_at DESC
                 LIMIT -1 OFFSET 64)''')
        return True

    def cached_proposal(self, key: Optional[str], scope: str) -> Optional[dict]:
        if not key:
            return None
        row = self.db.execute('''SELECT key,answer,skill_id FROM verified_proposals
            WHERE key=? AND scope=? AND active=1''', (key, scope)).fetchone()
        return dict(row) if row else None

    def replay_result(self, key: str, passed: bool):
        row = self.db.execute('SELECT skill_id FROM verified_proposals WHERE key=?',
                              (key,)).fetchone()
        if row is None:
            return
        with self.db:
            if passed:
                self.db.execute('''UPDATE verified_proposals SET uses=uses+1, updated_at=?
                    WHERE key=?''', (time.time(), key))
                if row['skill_id']:
                    self.db.execute('''UPDATE attention_memory SET successes=successes+1,
                        salience=MIN(10,salience+1), updated_at=? WHERE id=?''',
                        (time.time(), row['skill_id']))
            else:
                self.db.execute('''UPDATE verified_proposals SET active=0, updated_at=?
                    WHERE key=?''', (time.time(), key))
                if row['skill_id']:
                    self.db.execute('''UPDATE attention_memory SET failures=failures+1,
                        salience=MAX(0,salience-2),
                        status=CASE WHEN failures+1>=2 THEN 'retired' ELSE status END,
                        updated_at=? WHERE id=?''', (time.time(), row['skill_id']))
