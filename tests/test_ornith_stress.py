from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_verified_repairs import load_cases
from stress_ornith_reasoning import assess_response, normalize_answer, prompt_for, schema_for, score_answer


class OrnithStressTests(unittest.TestCase):
    def setUp(self):
        self.cases = load_cases(ROOT / 'experiments/verified-reasoning/challenges.jsonl')

    def test_all_challenge_answers_are_machine_checked(self):
        self.assertEqual({row['verifier']['kind'] for row in self.cases},
                         {'arithmetic', 'logic', 'plan'})
        for row in self.cases:
            self.assertIsNone(score_answer(row, row['response']['answer']))
        self.assertEqual(self.cases[3]['response']['answer'], '6356')

    def test_wrong_answers_and_invalid_plan_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incorrect'):
            score_answer(self.cases[0], '1/2')
        with self.assertRaisesRegex(ValueError, 'truth table'):
            score_answer(self.cases[1], 'not_entailed')
        with self.assertRaisesRegex(ValueError, 'precondition'):
            score_answer(self.cases[2], ['deploy'])

    def test_prompt_hides_oracle_and_includes_feedback(self):
        row = self.cases[2]
        prompt = prompt_for(row, [{'answer': ['deploy'], 'feedback': 'Plan violates an action precondition'}])
        self.assertIn('Previous attempts and checker feedback', prompt)
        self.assertIn('patch_deployed', prompt)
        self.assertIn('JSON array of action-name strings', prompt)
        self.assertNotIn('diagnose","snapshot","patch","deploy"', prompt)
        self.assertEqual(schema_for(row)['json_schema']['schema']['properties']['answer']['type'],
                         'array')

    def test_semantic_equivalents_keep_format_failure_visible(self):
        plan, valid = normalize_answer(self.cases[2],
                                       'diagnose, snapshot, patch, deploy, verify, audit, close')
        self.assertFalse(valid)
        self.assertIsNone(score_answer(self.cases[2], plan))
        logic, valid = normalize_answer(self.cases[1], 'Yes')
        self.assertEqual(logic, 'entailed')
        self.assertFalse(valid)

    def test_fenced_valid_plan_is_accepted_without_format_retry(self):
        content = '```json\n{"answer":"diagnose, snapshot, patch, deploy, verify, audit, close"}\n```'
        result = assess_response(self.cases[2], content)
        self.assertTrue(result['passed'])
        self.assertFalse(result['format_valid'])
        self.assertEqual(len(result['normalized_answer']), 7)

    def test_exact_calculator_accepts_valid_expression_only(self):
        row = self.cases[0]
        result = assess_response(row, '{"expression":"2*(1/6+1/4-1/12)+(1/6-1/12)"}',
                                 tool_assisted=True)
        self.assertTrue(result['passed'])
        self.assertEqual(result['normalized_answer'], '3/4')
        wrong = assess_response(row, '{"expression":"2*(1/6+1/4-1/12)"}',
                                tool_assisted=True)
        self.assertFalse(wrong['passed'])
        self.assertIn('wrong result', wrong['feedback'])
        alias = assess_response(row,
                                '```json\n{"answer":"2*(1/6+1/4-1/12)+(1/6-1/12)"}\n```',
                                tool_assisted=True)
        self.assertTrue(alias['passed'])
        self.assertFalse(alias['format_valid'])
        prose = assess_response(row, '{"expression":"The pumps fill half the tank"}',
                                tool_assisted=True)
        self.assertFalse(prose['passed'])
        self.assertIn('syntax', prose['feedback'])

    def test_counting_error_is_distinguished_from_combination_alias(self):
        row = self.cases[3]
        content = '{"expression":"com(8,3)*com(9,3)+com(8,4)*com(9,2)"}'
        result = assess_response(row, content, tool_assisted=True)
        self.assertFalse(result['passed'])
        self.assertIn('wrong result', result['feedback'])
        self.assertIn('incompatible people', result['feedback'])


if __name__ == '__main__':
    unittest.main()
