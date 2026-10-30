from pathlib import Path
import tempfile
import unittest

from xing_specialist.workspace import PatchError, apply_edits, apply_patch, validate_patch


PATCH = '''diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1 +1 @@
-def add(a, b): return a - b
+def add(a, b): return a + b
'''


class WorkspaceTests(unittest.TestCase):
    def test_approved_patch_applies(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            path = Path(folder) / 'calc.py'
            path.write_text('def add(a, b): return a - b\n')
            self.assertEqual(apply_patch(Path(folder), PATCH, {'calc.py'}), ('calc.py',))
            self.assertIn('a + b', path.read_text())

    def test_unapproved_and_traversal_paths_rejected(self):
        with self.assertRaises(PatchError):
            validate_patch(PATCH, {'other.py'})
        with self.assertRaises(PatchError):
            validate_patch(PATCH.replace('a/calc.py b/calc.py', 'a/../outside.py b/../outside.py'),
                           {'../outside.py'})
        with self.assertRaises(PatchError):
            validate_patch(PATCH.replace('+++ b/calc.py', '+++ b/other.py'), {'calc.py'})

    def test_exact_edit_requires_unique_source(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'artifacts') as folder:
            path = Path(folder) / 'calc.py'
            path.write_text('return a - b\n')
            edit = [{'path': 'calc.py', 'old': 'a - b', 'new': 'a + b'}]
            self.assertEqual(apply_edits(Path(folder), edit, {'calc.py'}), ('calc.py',))
            with self.assertRaises(PatchError):
                apply_edits(Path(folder), edit, {'calc.py'})


if __name__ == '__main__':
    unittest.main()
