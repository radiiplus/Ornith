import json
from pathlib import Path
import tempfile
import threading
import unittest

from xing_specialist.agent import build_model_prompt
from xing_specialist.backend import GenerationLimits
from xing_specialist.deliberation import deliberate
from xing_specialist.memory import MemoryStore


class ReasoningBackend:
    def __init__(self):
        self.analyst_threads = set()
        self.barrier = threading.Barrier(2, timeout=3)

    def generate(self, messages, limits):
        system = messages[0].content
        if 'Decompose' in system:
            return json.dumps({'units': [
                {'task': 'Inspect operator', 'done_when': 'Operator matches addition'},
                {'task': 'Inspect tests', 'done_when': 'A test fails before the edit'}]})
        if 'independent repair analyst' in system:
            self.analyst_threads.add(threading.get_ident())
            self.barrier.wait()
            return json.dumps({'finding': 'Operator is subtraction',
                               'evidence': 'Source shows a - b', 'uncertainty': 'Tests may differ'})
        if 'Review independent' in system:
            return json.dumps({'contradictions': ['Test claim is not established'],
                               'guidance': 'Check the baseline test result'})
        raise AssertionError(system)


class DeliberationTests(unittest.TestCase):
    def test_parallel_units_are_recorded_and_reviewed(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / 'memory.sqlite3')
            task_id = store.task('Fix add')
            backend = ReasoningBackend()
            notes = deliberate(backend, GenerationLimits(), store, task_id,
                               'Fix add', 'def add(a,b): return a-b', 'Baseline failed', 2)
            self.assertEqual(len(backend.analyst_threads), 2)
            self.assertIn('Test claim is not established', notes)
            prompt, _ = build_model_prompt('Fix add', 'def add(a,b): return a-b',
                                           'Baseline failed', [], ('calc.py',),
                                           GenerationLimits(), working_notes=notes)
            self.assertIn('Test claim is not established', prompt)
            rows = store.db.execute('SELECT status FROM reasoning_units WHERE task_id=?',
                                    (task_id,)).fetchall()
            self.assertEqual([row['status'] for row in rows], ['analyzed', 'analyzed'])
            self.assertEqual(len(store.progress(task_id)['reasoning_units']), 2)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM actions WHERE kind='reasoning_review'").fetchone()[0], 1)
            store.close()

    def test_invalid_plan_falls_back_without_units(self):
        class InvalidBackend:
            def generate(self, messages, limits):
                return '{"units":[]}'

        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / 'memory.sqlite3')
            task_id = store.task('Fix add')
            notes = deliberate(InvalidBackend(), GenerationLimits(), store, task_id,
                               'Fix add', 'source', 'baseline', 1)
            self.assertEqual(notes, '')
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM reasoning_units').fetchone()[0], 0)
            store.close()
