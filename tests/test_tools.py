from pathlib import Path
import tempfile
import unittest

from xing_specialist.tools import CommandPolicy, IsolatedExecutor, ToolError


class ToolTests(unittest.TestCase):
    def test_policy_rejects_shell_and_python_code(self):
        for command in [('bash', '-c', 'true'), ('python3', '-c', 'print(1)'), ('git', 'reset', '--hard')]:
            with self.assertRaises(ToolError):
                CommandPolicy.approve(command)

    def test_isolated_test_and_workspace_copy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            root = Path(folder)
            repo = root / 'repo'
            (repo / 'tests').mkdir(parents=True)
            (repo / 'tests' / 'test_sample.py').write_text(
                'import pathlib, unittest\n'
                'class Check(unittest.TestCase):\n'
                '  def test_isolation(self):\n'
                '    self.assertFalse(pathlib.Path("/home/ec2-user/.aws").exists())\n'
                '    pathlib.Path("generated.txt").write_text("sandbox")\n')
            executor = IsolatedExecutor(root / 'state', timeout_seconds=10, memory_mib=512)
            result = executor.run(repo, ('python3', '-m', 'unittest', 'discover', '-s', 'tests'))
            self.assertEqual(result.exit_code, 0, result.stderr)
            self.assertIn('Ran 1 test', result.stderr)
            self.assertFalse((repo / 'generated.txt').exists())

    def test_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            root = Path(folder)
            repo = root / 'repo'
            (repo / 'tests').mkdir(parents=True)
            (repo / 'tests' / 'test_sleep.py').write_text(
                'import time, unittest\nclass Check(unittest.TestCase):\n'
                '  def test_sleep(self): time.sleep(5)\n')
            executor = IsolatedExecutor(root / 'state', timeout_seconds=1, memory_mib=512)
            result = executor.run(repo, ('python3', '-m', 'unittest', 'discover', '-s', 'tests'))
            self.assertTrue(result.timed_out)
            self.assertEqual(result.exit_code, 124)


if __name__ == '__main__':
    unittest.main()
