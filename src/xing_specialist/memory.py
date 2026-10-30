"""SQLite audit trail and evidence-backed lessons."""

from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3


def redact(value: str) -> str:
    value = re.sub(r'github_pat_[A-Za-z0-9_]+', '[REDACTED_GITHUB_TOKEN]', value)
    value = re.sub(r'AKIA[0-9A-Z]{16}', '[REDACTED_AWS_KEY]', value)
    value = re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]+', r'\1[REDACTED]', value)
    value = re.sub(r'(?i)((?:token|secret|password|api[_-]?key)\s*[=:]\s*)\S+', r'\1[REDACTED]', value)
    value = re.sub(r'(?i)("(?:token|secret|password|api[_-]?key)"\s*:\s*")[^"]+',
                   r'\1[REDACTED]', value)
    return value[:65536]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys = ON')
        self.db.execute('PRAGMA journal_mode = WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY, objective TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES tasks(id),
                claim TEXT NOT NULL, expected TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS plan_steps (
                task_id INTEGER NOT NULL REFERENCES tasks(id), key TEXT NOT NULL,
                expected TEXT NOT NULL, requires TEXT NOT NULL, status TEXT NOT NULL,
                PRIMARY KEY(task_id,key));
            CREATE TABLE IF NOT EXISTS reasoning_units (
                task_id INTEGER NOT NULL REFERENCES tasks(id), key TEXT NOT NULL,
                task TEXT NOT NULL, done_when TEXT NOT NULL, status TEXT NOT NULL,
                finding TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
                PRIMARY KEY(task_id,key));
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES tasks(id),
                hypothesis_id INTEGER REFERENCES hypotheses(id), kind TEXT NOT NULL,
                detail TEXT NOT NULL, stdout TEXT NOT NULL, stderr TEXT NOT NULL,
                exit_code INTEGER, duration_seconds REAL NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES tasks(id),
                action_id INTEGER REFERENCES actions(id), body TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS parked_observations (
                id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES tasks(id),
                observation TEXT NOT NULL, context TEXT NOT NULL, relevance_tag TEXT NOT NULL,
                reason TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES tasks(id),
                action_id INTEGER REFERENCES actions(id), command TEXT NOT NULL,
                expected_exit INTEGER NOT NULL, actual_exit INTEGER, passed INTEGER NOT NULL,
                regression INTEGER NOT NULL, stdout TEXT NOT NULL, stderr TEXT NOT NULL,
                created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY, task_id INTEGER NOT NULL REFERENCES tasks(id),
                body TEXT NOT NULL, evidence_test_id INTEGER NOT NULL REFERENCES test_results(id),
                created_at TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS lesson_fts USING fts5(body, content='lessons', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS lesson_insert AFTER INSERT ON lessons BEGIN
                INSERT INTO lesson_fts(rowid, body) VALUES (new.id, new.body);
            END;
        ''')
        self.db.commit()

    def close(self):
        self.db.close()

    def task(self, objective: str) -> int:
        stamp = now()
        with self.db:
            cursor = self.db.execute('INSERT INTO tasks(objective,status,created_at,updated_at) VALUES (?,?,?,?)',
                                     (redact(objective), 'pending', stamp, stamp))
        return cursor.lastrowid

    def set_status(self, task_id: int, status: str):
        if status not in ('pending', 'running', 'verified', 'failed', 'needs_review'):
            raise ValueError('Invalid task status')
        with self.db:
            self.db.execute('UPDATE tasks SET status=?, updated_at=? WHERE id=?', (status, now(), task_id))

    def hypothesis(self, task_id: int, claim: str, expected: str) -> int:
        with self.db:
            cursor = self.db.execute('INSERT INTO hypotheses(task_id,claim,expected,created_at) VALUES (?,?,?,?)',
                                     (task_id, redact(claim), redact(expected), now()))
        return cursor.lastrowid

    def add_plan(self, task_id: int, steps: tuple):
        with self.db:
            for step in steps:
                self.db.execute('INSERT INTO plan_steps(task_id,key,expected,requires,status) VALUES (?,?,?,?,?)',
                                (task_id, step.key, redact(step.expected), ','.join(step.requires), 'pending'))

    def set_step_status(self, task_id: int, key: str, status: str):
        if status not in ('pending', 'running', 'verified', 'failed'):
            raise ValueError('Invalid plan-step status')
        with self.db:
            self.db.execute('UPDATE plan_steps SET status=? WHERE task_id=? AND key=?',
                            (status, task_id, key))

    def add_reasoning_units(self, task_id: int, units: tuple):
        with self.db:
            for unit in units:
                self.db.execute('''INSERT INTO reasoning_units
                    (task_id,key,task,done_when,status,finding,updated_at)
                    VALUES (?,?,?,?,'pending','',?)''',
                    (task_id, unit.key, redact(unit.task), redact(unit.done_when), now()))

    def update_reasoning_unit(self, task_id: int, key: str, status: str, finding: str):
        if status not in ('analyzed', 'failed'):
            raise ValueError('Invalid reasoning-unit status')
        with self.db:
            self.db.execute('''UPDATE reasoning_units SET status=?,finding=?,updated_at=?
                WHERE task_id=? AND key=?''', (status, redact(finding), now(), task_id, key))

    def progress(self, task_id: int) -> dict:
        stages = self.db.execute('''SELECT key,status FROM plan_steps WHERE task_id=? ORDER BY rowid''',
                                 (task_id,)).fetchall()
        units = self.db.execute('''SELECT key,task,done_when,status FROM reasoning_units
            WHERE task_id=? ORDER BY key''', (task_id,)).fetchall()
        return {'stages': [dict(row) for row in stages],
                'reasoning_units': [dict(row) for row in units]}

    def action(self, task_id: int, hypothesis_id: int, kind: str, detail: str,
               stdout: str = '', stderr: str = '', exit_code=None, duration_seconds: float = 0) -> int:
        with self.db:
            cursor = self.db.execute('''INSERT INTO actions
                (task_id,hypothesis_id,kind,detail,stdout,stderr,exit_code,duration_seconds,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)''',
                (task_id, hypothesis_id, kind, redact(detail), redact(stdout), redact(stderr),
                 exit_code, duration_seconds, now()))
        return cursor.lastrowid

    def observation(self, task_id: int, action_id: int, body: str) -> int:
        with self.db:
            cursor = self.db.execute('INSERT INTO observations(task_id,action_id,body,created_at) VALUES (?,?,?,?)',
                                     (task_id, action_id, redact(body), now()))
        return cursor.lastrowid

    def park_observation(self, task_id: int, observation: str, context: str,
                         relevance_tag: str, reason: str) -> int:
        if not all(isinstance(value, str) and value.strip()
                   for value in (observation, context, relevance_tag, reason)):
            raise ValueError('Parked observation fields must be nonempty strings')
        with self.db:
            cursor = self.db.execute('''INSERT INTO parked_observations
                (task_id,observation,context,relevance_tag,reason,created_at)
                VALUES (?,?,?,?,?,?)''',
                (task_id, redact(observation)[:1000], redact(context)[:500],
                 redact(relevance_tag)[:120], redact(reason)[:300], now()))
        return cursor.lastrowid

    def recall_parked(self, objective: str, limit: int = 5) -> list:
        words = set(re.findall(r'[a-z0-9_]{4,}', objective.lower())) - {
            'that', 'this', 'with', 'from', 'when', 'into', 'then', 'test', 'tests',
            'code', 'file', 'fix', 'repair', 'find', 'would', 'should', 'after'}
        if not words:
            return []
        rows = self.db.execute('''SELECT id,observation,context,relevance_tag,reason
            FROM parked_observations ORDER BY id DESC LIMIT 500''').fetchall()
        ranked = []
        for row in rows:
            tag_words = set(re.findall(r'[a-z0-9_]{4,}', row['relevance_tag'].lower()))
            body_words = set(re.findall(r'[a-z0-9_]{4,}',
                                            (row['observation'] + ' ' + row['reason']).lower()))
            score = 2 * len(words & tag_words) + len(words & body_words)
            if score:
                ranked.append((score, row['id'], dict(row)))
        ranked.sort(reverse=True)
        return [item[2] for item in ranked[:max(0, min(limit, 10))]]

    def test_result(self, task_id: int, action_id: int, command: str, expected_exit: int,
                    actual_exit, stdout: str, stderr: str, regression: bool = False) -> int:
        passed = int(actual_exit == expected_exit)
        with self.db:
            cursor = self.db.execute('''INSERT INTO test_results
                (task_id,action_id,command,expected_exit,actual_exit,passed,regression,stdout,stderr,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',
                (task_id, action_id, redact(command), expected_exit, actual_exit, passed,
                 int(regression), redact(stdout), redact(stderr), now()))
        return cursor.lastrowid

    def lesson(self, task_id: int, body: str, evidence_test_id: int) -> int:
        row = self.db.execute('SELECT task_id,passed FROM test_results WHERE id=?', (evidence_test_id,)).fetchone()
        if row is None or row['task_id'] != task_id or not row['passed']:
            raise ValueError('A verified lesson requires passing test evidence from this task')
        with self.db:
            cursor = self.db.execute('INSERT INTO lessons(task_id,body,evidence_test_id,created_at) VALUES (?,?,?,?)',
                                     (task_id, redact(body), evidence_test_id, now()))
        return cursor.lastrowid
