import copy
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_verified_repairs import load_cases, prepare


class VerifiedRepairDataTests(unittest.TestCase):
    def setUp(self):
        self.seed = [json.loads(line) for line in
                     (ROOT / 'experiments/verified-repairs/seed.jsonl').read_text().splitlines()]

    def write_cases(self, folder, rows):
        source = folder / 'input.jsonl'
        source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return source

    def test_seed_requires_failing_baseline_and_passing_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output'
            counts = prepare(ROOT / 'experiments/verified-repairs/seed.jsonl', output)
            self.assertEqual(counts, {'train': 2, 'eval': 1})
            self.assertEqual(len((output / 'train.jsonl').read_text().splitlines()), 2)
            self.assertEqual(len((output / 'eval.jsonl').read_text().splitlines()), 1)
            self.assertTrue(all(row['baseline_exit'] != 0 and row['verified_exit'] == 0
                                for row in json.loads((output / 'manifest.json').read_text())))

    def test_group_leakage_is_rejected(self):
        rows = copy.deepcopy(self.seed)
        rows[2]['group'] = rows[0]['group']
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'crosses train and eval'):
                load_cases(self.write_cases(Path(folder), rows))

    def test_invalid_edit_does_not_create_output(self):
        rows = copy.deepcopy(self.seed)
        rows[0]['proposal']['edits'][0]['new'] = 'return a * b'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, 'did not pass'):
                prepare(self.write_cases(root, rows), root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_path_like_id_is_rejected(self):
        rows = copy.deepcopy(self.seed)
        rows[0]['id'] = '../escape'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Invalid id'):
                load_cases(self.write_cases(Path(folder), rows))


if __name__ == '__main__':
    unittest.main()
