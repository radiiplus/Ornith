import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from run_dual_pipeline import (SwappingModelProvider, check_verified_answer,
                               verified_context)  # noqa: E402
from xing_specialist.backend import GenerationLimits  # noqa: E402


class VerifiedContextTests(unittest.TestCase):
    def test_arithmetic_answer_is_not_exposed(self):
        row = {'verifier': {'kind': 'arithmetic', 'expression': '2*(1/6+1/4-1/12)'}}
        self.assertEqual(verified_context(row), {})

    def test_logic_and_plan_inputs_are_exposed(self):
        logic = {'verifier': {'kind': 'logic', 'variables': ['a'],
                              'premises': ['a'], 'conclusion': 'a'}}
        plan = {'verifier': {'kind': 'plan', 'initial': ['ready'],
                             'goal': ['done'], 'actions': {'finish': {'requires': ['ready']}}}}
        self.assertEqual(verified_context(logic), {
            'variables': ['a'], 'premises': ['a'], 'conclusion': 'a'})
        self.assertEqual(verified_context(plan), {
            'initial': ['ready'], 'goal': ['done'],
            'actions': {'finish': {'requires': ['ready']}}})

    def test_arithmetic_expression_is_checked_without_exposing_expected_expression(self):
        row = {'verifier': {'kind': 'arithmetic', 'expression': '2*(1/6+1/4-1/12)+(1/6-1/12)'}}
        self.assertEqual(verified_context(row), {})
        check_verified_answer(row, '2*(1/6+1/4-1/12)+(1/6-1/12)')
        with self.assertRaises(ValueError):
            check_verified_answer(row, '2*(1/6+1/4-1/12)')
        with self.assertRaises(SyntaxError):
            check_verified_answer(row, 'The pumps fill at a rate of 1/2 per hour')

    def test_backend_reset_reports_server_exit_and_guard(self):
        provider = SwappingModelProvider()
        provider.engine = SimpleNamespace(process=SimpleNamespace(poll=lambda: -11))
        provider.monitor = SimpleNamespace(breach_reason=None)
        provider.port = 1234
        with patch.object(provider, '_load', return_value=0), patch(
                'run_dual_pipeline.OpenAICompatibleBackend.generate',
                side_effect=ConnectionResetError('reset')):
            with self.assertRaisesRegex(RuntimeError, 'server_exit=-11; memory_guard=None'):
                provider.generate('execution', [], GenerationLimits())


if __name__ == '__main__':
    unittest.main()
