import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_verified_repairs import load_cases, prepare


class VerifiedReasoningDataTests(unittest.TestCase):
    def setUp(self):
        self.seed_path = ROOT / 'experiments/verified-reasoning/seed.jsonl'
        self.seed = [json.loads(line) for line in self.seed_path.read_text().splitlines()]

    def write_cases(self, folder, rows):
        path = folder / 'cases.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def test_mixed_seed_is_checked_and_grouped(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output'
            self.assertEqual(prepare(self.seed_path, output), {'train': 5, 'eval': 4})
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual({row['domain'] for row in manifest},
                             {'code_repair', 'quantitative', 'planning', 'logic'})
            train_groups = {row['group'] for row in manifest if row['split'] == 'train'}
            eval_groups = {row['group'] for row in manifest if row['split'] == 'eval'}
            self.assertFalse(train_groups & eval_groups)
            self.assertTrue(all(row['answer_checked'] for row in manifest
                                if row['domain'] != 'code_repair'))

    def test_wrong_arithmetic_answer_is_rejected(self):
        rows = copy.deepcopy(self.seed)
        rows[3]['response']['answer'] = '25'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Arithmetic answer'):
                load_cases(self.write_cases(Path(folder), rows))

    def test_plan_with_missing_precondition_is_rejected(self):
        rows = copy.deepcopy(self.seed)
        rows[5]['response']['answer'] = ['send', 'validate', 'draft']
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'precondition'):
                load_cases(self.write_cases(Path(folder), rows))

    def test_false_entailment_is_rejected(self):
        rows = copy.deepcopy(self.seed)
        rows[8]['response']['answer'] = 'entailed'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'truth table'):
                load_cases(self.write_cases(Path(folder), rows))

    def test_bad_expression_cannot_execute_code(self):
        rows = copy.deepcopy(self.seed)
        rows[3]['verifier']['expression'] = '__import__("os").system("id")'
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Unsupported arithmetic'):
                load_cases(self.write_cases(Path(folder), rows))


if __name__ == '__main__':
    unittest.main()
