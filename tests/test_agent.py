from pathlib import Path
import json
import tempfile
import unittest

from xing_specialist.agent import EngineeringAgent, build_model_prompt, parse_proposal
from xing_specialist.backend import BackendError, GenerationLimits
from xing_specialist.memory import MemoryStore
from xing_specialist.tools import IsolatedExecutor
from xing_specialist.workspace import PatchError


PATCH = '''diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1 +1 @@
-def add(a, b): return a - b
+def add(a, b): return a + b
'''


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, limits):
        self.calls += 1
        return json.dumps({'hypothesis': 'The operator is wrong',
                           'expected': 'Addition test passes', 'patch': PATCH})


class FailedBackend:
    def generate(self, messages, limits):
        raise BackendError('Model server returned HTTP 503')


class AgentTests(unittest.TestCase):
    def test_research_reference_reaches_repair_prompt_as_unverified_source(self):
        prompt, _ = build_model_prompt('Fix add', 'def add(a,b): return a-b',
                                       'Baseline failed', [], ('calc.py',),
                                       GenerationLimits(), research_refs=[{
                                           'kind': 'procedure', 'title': 'Addition check',
                                           'quote': 'Test add(2,3)',
                                           'locator': 'https://example.com/addition'}])
        self.assertIn('Test add(2,3)', prompt)
        self.assertIn('not verified facts', prompt)

    def test_fenced_json_from_temporary_model(self):
        self.assertEqual(parse_proposal('```json\n{"hypothesis":"x"}\n```')['hypothesis'], 'x')

    def test_staged_repair_and_optional_promotion(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            root = Path(folder)
            repo = root / 'repo'
            (repo / 'tests').mkdir(parents=True)
            (repo / 'calc.py').write_text('def add(a, b): return a - b\n')
            (repo / 'tests' / 'test_calc.py').write_text(
                'import unittest\nfrom calc import add\n'
                'class Check(unittest.TestCase):\n'
                '  def test_add(self): self.assertEqual(add(2, 3), 5)\n')
            store = MemoryStore(root / 'memory.sqlite3')
            executor = IsolatedExecutor(root / 'sandboxes', timeout_seconds=10, memory_mib=512)
            agent = EngineeringAgent(FakeBackend(), store, executor, GenerationLimits(2048, 512, 5))
            commands = (('python3', '-m', 'unittest', 'discover', '-s', 'tests'),)
            staged = agent.run('Fix add', repo, ('calc.py',), commands)
            self.assertEqual(staged.status, 'needs_review')
            self.assertIn('a - b', (repo / 'calc.py').read_text())
            promoted = agent.run('Fix add', repo, ('calc.py',), commands, apply_to_source=True)
            self.assertEqual(promoted.status, 'verified')
            self.assertIn('a + b', (repo / 'calc.py').read_text())
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM lessons').fetchone()[0], 2)
            store.close()

    def test_verified_proposal_replay_checks_tests_and_snapshot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            root = Path(folder)
            repo = root / 'repo'
            (repo / 'tests').mkdir(parents=True)
            (repo / 'calc.py').write_text('def add(a, b): return a - b\n')
            test_file = repo / 'tests' / 'test_calc.py'
            test_file.write_text('import unittest\nfrom calc import add\n'
                                 'class Check(unittest.TestCase):\n'
                                 '  def test_add(self): self.assertEqual(add(2, 3), 5)\n')
            backend = FakeBackend()
            store = MemoryStore(root / 'memory.sqlite3')
            executor = IsolatedExecutor(root / 'sandboxes', timeout_seconds=10, memory_mib=512)
            agent = EngineeringAgent(backend, store, executor, GenerationLimits(2048, 512, 5))
            commands = (('python3', '-m', 'unittest', 'discover', '-s', 'tests'),)
            try:
                self.assertEqual(agent.run('Fix add', repo, ('calc.py',), commands).status, 'needs_review')
                self.assertEqual(backend.calls, 1)
                replay = agent.run('Fix add', repo, ('calc.py',), commands)
                self.assertEqual(replay.status, 'needs_review')
                self.assertEqual(backend.calls, 1)
                self.assertEqual(store.db.execute(
                    "SELECT COUNT(*) FROM actions WHERE task_id=? AND kind='verification'",
                    (replay.task_id,)).fetchone()[0], 2)
                self.assertEqual(store.db.execute(
                    "SELECT COUNT(*) FROM actions WHERE task_id=? AND kind='proposal_replay'",
                    (replay.task_id,)).fetchone()[0], 1)
                test_file.write_text(test_file.read_text() + '\n# changed test input\n')
                self.assertEqual(agent.run('Fix add', repo, ('calc.py',), commands).status, 'needs_review')
                self.assertEqual(backend.calls, 2)
            finally:
                store.close()

    def test_backend_failure_is_audited(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            root = Path(folder)
            repo = root / 'repo'
            (repo / 'tests').mkdir(parents=True)
            (repo / 'calc.py').write_text('def add(a, b): return a - b\n')
            (repo / 'tests' / 'test_calc.py').write_text('import unittest\nclass Check(unittest.TestCase):\n  def test_ok(self): self.assertTrue(True)\n')
            store = MemoryStore(root / 'memory.sqlite3')
            agent = EngineeringAgent(FailedBackend(), store, IsolatedExecutor(root / 'sandboxes'))
            result = agent.run('Fix add', repo, ('calc.py',),
                               (('python3', '-m', 'unittest', 'discover', '-s', 'tests'),))
            self.assertEqual(result.status, 'failed')
            self.assertEqual(store.db.execute('SELECT status FROM tasks WHERE id=?',
                                              (result.task_id,)).fetchone()[0], 'failed')
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM actions WHERE kind='backend_failure'").fetchone()[0], 1)
            store.close()

    def test_passing_baseline_never_promotes_or_creates_lesson(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            root = Path(folder)
            repo = root / 'repo'
            (repo / 'tests').mkdir(parents=True)
            (repo / 'calc.py').write_text('def add(a, b): return a - b\n')
            (repo / 'tests' / 'test_calc.py').write_text('import unittest\nclass Check(unittest.TestCase):\n  def test_ok(self): self.assertTrue(True)\n')
            store = MemoryStore(root / 'memory.sqlite3')
            agent = EngineeringAgent(FakeBackend(), store, IsolatedExecutor(root / 'sandboxes'))
            result = agent.run('Fix add', repo, ('calc.py',),
                               (('python3', '-m', 'unittest', 'discover', '-s', 'tests'),),
                               apply_to_source=True)
            self.assertEqual(result.status, 'needs_review')
            self.assertIn('a - b', (repo / 'calc.py').read_text())
            self.assertEqual(store.db.execute('SELECT COUNT(*) FROM lessons').fetchone()[0], 0)
            store.close()


if __name__ == '__main__':
    unittest.main()
