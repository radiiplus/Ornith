from pathlib import Path
import tempfile
import unittest

from xing_specialist.adaptive import AdaptiveMemory, repository_snapshot_key
from xing_specialist.memory import MemoryStore


class AdaptiveMemoryTests(unittest.TestCase):
    def test_repeated_gap_verified_skill_revision_and_replay_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(Path(folder) / 'memory.sqlite3')
            memory = AdaptiveMemory(store, max_items=4)
            try:
                scope = '/work/example'
                topic = 'Solve multi step planning'
                memory.observe_gap(scope, topic, 'verification', 'Wrong final constraint')
                self.assertEqual(memory.recall(scope, topic), [])
                memory.observe_gap(scope, topic, 'verification', 'Wrong final constraint again')
                self.assertEqual(memory.recall(scope, topic)[0]['kind'], 'gap')
                self.assertEqual(memory.recall(scope, 'Fix renderer layout'), [])
                task = store.task(topic)
                failed = store.test_result(task, None, 'checker', 0, 1, '', '')
                with self.assertRaises(ValueError):
                    memory.remember_skill(scope, topic, 'Check final constraint', failed)
                passed = store.test_result(task, None, 'checker', 0, 0, '', '')
                skill = memory.remember_skill(scope, topic, 'Check final constraint', passed)
                self.assertEqual(memory.recall(scope, topic)[0]['kind'], 'skill')
                memory.resolve_gaps(scope, topic)
                self.assertEqual(len(memory.recall(scope, topic)), 1)
                self.assertTrue(memory.store_proposal('snapshot', scope, '{"answer":42}', passed, skill))
                self.assertIsNotNone(memory.cached_proposal('snapshot', scope))
                memory.replay_result('snapshot', False)
                self.assertIsNone(memory.cached_proposal('snapshot', scope))
                memory.remember_skill(scope, topic, 'Check final constraint', passed)
                self.assertEqual(store.db.execute('SELECT failures FROM attention_memory WHERE id=?',
                                                  (skill,)).fetchone()[0], 1)
                memory.store_proposal('snapshot', scope, '{"answer":42}', passed, skill)
                memory.replay_result('snapshot', False)
                self.assertEqual(store.db.execute('SELECT status FROM attention_memory WHERE id=?',
                                                  (skill,)).fetchone()[0], 'retired')
                self.assertEqual(memory.recall(scope, topic), [])
                memory.remember_skill(scope, topic, 'Check every final constraint', passed)
                row = store.db.execute('SELECT version,failures,body FROM attention_memory WHERE id=?',
                                       (skill,)).fetchone()
                self.assertEqual(row['version'], 2)
                self.assertEqual(row['failures'], 0)
                self.assertIn('every', row['body'])
            finally:
                store.close()

    def test_snapshot_changes_with_relevant_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            (repo / 'target.py').write_text('value = 1\n')
            commands = (('python3', '-m', 'unittest'),)
            before = repository_snapshot_key(repo, 'Fix value', ('target.py',), commands, ())
            self.assertEqual(before, repository_snapshot_key(repo, 'Fix value', ('target.py',), commands, ()))
            (repo / 'target.py').write_text('value = 2\n')
            self.assertNotEqual(before, repository_snapshot_key(repo, 'Fix value', ('target.py',), commands, ()))
            self.assertNotEqual(before, repository_snapshot_key(repo, 'Fix another value',
                                                                 ('target.py',), commands, ()))


if __name__ == '__main__':
    unittest.main()
