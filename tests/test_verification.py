from pathlib import Path
import tempfile
import unittest

from xing_specialist.memory import MemoryStore
from xing_specialist.tools import CommandResult
from xing_specialist.verification import Verifier


class FakeExecutor:
    def __init__(self, exit_code):
        self.exit_code = exit_code

    def run(self, repository, command):
        return CommandResult(tuple(command), '', 'failure' if self.exit_code else '',
                             self.exit_code, 0.01)


class VerificationTests(unittest.TestCase):
    def test_observed_regression_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            store = MemoryStore(Path(folder) / 'memory.sqlite3')
            task = store.task('Fix a test')
            hypothesis = store.hypothesis(task, 'Cause', 'Expected')
            command = ('python3', '-m', 'unittest')
            baseline = CommandResult(command, 'ok', '', 0, 0.01)
            report = Verifier(FakeExecutor(1), store).verify(task, hypothesis, folder,
                                                             (command,), (baseline,))
            self.assertFalse(report.passed)
            self.assertEqual(report.regressions, 1)
            row = store.db.execute('SELECT passed,regression FROM test_results WHERE id=?',
                                   (report.evidence_ids[0],)).fetchone()
            self.assertEqual(tuple(row), (0, 1))
            store.close()

    def test_passing_test_creates_eligible_evidence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            store = MemoryStore(Path(folder) / 'memory.sqlite3')
            task = store.task('Fix a test')
            hypothesis = store.hypothesis(task, 'Cause', 'Expected')
            command = ('python3', '-m', 'unittest')
            report = Verifier(FakeExecutor(0), store).verify(task, hypothesis, folder, (command,))
            self.assertTrue(report.passed)
            self.assertGreater(store.lesson(task, 'Verified repair', report.evidence_ids[0]), 0)
            store.close()


if __name__ == '__main__':
    unittest.main()
