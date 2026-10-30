from pathlib import Path
import tempfile
import unittest

from xing_specialist.memory import MemoryStore, redact
from xing_specialist.planner import Planner
from xing_specialist.retrieval import LessonRetriever


class MemoryRetrievalTests(unittest.TestCase):
    def test_evidence_gate_persistence_and_search(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'memory.sqlite3'
            store = MemoryStore(path)
            task = store.task('Fix compiler parser')
            store.add_plan(task, Planner().plan('Fix compiler parser'))
            store.set_step_status(task, 'inspect', 'verified')
            hypothesis = store.hypothesis(task, 'A missing bounds check', 'Malformed input is rejected')
            action = store.action(task, hypothesis, 'test', 'python3 -m unittest', 'ok', '', 0, 0.2)
            failed = store.test_result(task, action, 'python3 -m unittest', 0, 1, '', 'failure')
            with self.assertRaises(ValueError):
                store.lesson(task, 'Parser bounds check worked', failed)
            passed = store.test_result(task, action, 'python3 -m unittest', 0, 0, 'ok', '')
            store.lesson(task, 'Parser bounds check prevents overread', passed)
            store.close()
            store = MemoryStore(path)
            self.assertEqual(store.db.execute("SELECT status FROM plan_steps WHERE task_id=? AND key='inspect'",
                                              (task,)).fetchone()[0], 'verified')
            found = LessonRetriever(store).search('parser bounds')
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]['evidence_test_id'], passed)
            store.close()

    def test_secrets_redacted_and_query_sanitized(self):
        self.assertNotIn('AKIA1234567890ABCDEF', redact('AKIA1234567890ABCDEF'))
        self.assertNotIn('privatevalue', redact('{"api_key":"privatevalue"}'))
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / 'memory.sqlite3')
            self.assertEqual(LessonRetriever(store).search('" OR *'), [])
            store.close()


if __name__ == '__main__':
    unittest.main()
