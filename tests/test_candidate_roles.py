import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate_candidate_roles import gate_prompt, shortest_plan_length  # noqa: E402
from prepare_verified_repairs import check_plan  # noqa: E402


class CandidateRoleTests(unittest.TestCase):
    def test_shortest_plan_respects_preconditions_and_goal(self):
        verifier = {'initial': ['ready'], 'goal': ['done'], 'actions': {
            'inspect': {'requires': ['ready'], 'adds': ['known'], 'removes': []},
            'fix': {'requires': ['known'], 'adds': ['done'], 'removes': []},
            'undo': {'requires': ['known'], 'adds': [], 'removes': ['known']}}}
        self.assertEqual(shortest_plan_length(verifier), 2)
        check_plan(verifier, ['inspect', 'fix'])

    def test_gate_prompt_contains_full_original_objective(self):
        objective = 'Passing tests is sufficient for deployment. Must deployment pass tests?'
        prompt = gate_prompt(objective)
        self.assertIn(objective, prompt)
        self.assertIn('Preserve the question and constraints', prompt)


if __name__ == '__main__':
    unittest.main()
