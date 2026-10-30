from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_ornith_bytecode as probe


class BytecodeProbeTests(unittest.TestCase):
    def test_fixture_has_verified_optimized_mode_bypass(self):
        result = probe.verify_fixture()
        self.assertTrue(result['verified'])
        self.assertEqual(result['optimized_output'], 'OUTSIDE_MARKER')

    def test_optimized_bytecode_omits_guard_but_reads_file(self):
        source = probe.FIXTURE.read_text()
        bytecode = probe.disassemble_optimized(source)
        self.assertNotIn('is_relative_to', bytecode)
        self.assertIn('read_text', bytecode)
        delta = probe.bytecode_delta(source)
        self.assertEqual(delta['assert_statements_in_source'], 1)
        self.assertIn('is_relative_to', delta['methods_removed_in_optimized'])
        self.assertIn('read_text', delta['methods_remaining_in_optimized'])

    def test_answer_scoring_requires_specific_cause(self):
        correct = ('{"verdict":"unsafe","cause":"assert_removed",'
                   '"evidence":"Optimized bytecode omits the assert guard."}')
        self.assertTrue(probe.assess_answer(correct)['passed'])
        observed = ('{"verdict":"unsafe","cause":"assert_removed",'
                    '"evidence":"The is_relative_to assertion is stripped under python -O."}')
        self.assertTrue(probe.assess_answer(observed)['passed'])
        self.assertFalse(probe.assess_answer(
            '{"verdict":"unsafe","cause":"assert_removed","evidence":"Looks fine."}')['passed'])
        self.assertIsNone(probe.parse_answer('not json')['cause'])

    def test_reasoned_answer_checks_predictions_and_cause(self):
        truth = probe.verify_fixture()
        answer = ('{"source_guard":"is_relative_to","optimized_guard":"absent",'
                  '"normal_result":"blocked","optimized_result":"reads_outside",'
                  '"cause":"assert_removed",'
                  '"evidence":"The assertion is omitted in optimized bytecode."}')
        self.assertTrue(probe.assess_reasoned_answer(answer, truth)['passed'])
        incorrect = answer.replace('"absent"', '"present"')
        self.assertFalse(probe.assess_reasoned_answer(incorrect, truth)['prediction_passed'])
        wrong_cause = answer.replace('"assert_removed"', '"injection"')
        self.assertFalse(probe.assess_reasoned_answer(wrong_cause, truth)['cause_passed'])

    def test_feedback_uses_checked_behavior_and_disassembly(self):
        feedback = probe.checker_feedback(
            {'answer': {'cause': 'injection'}}, probe.verify_fixture(),
            probe.bytecode_delta(probe.FIXTURE.read_text()))
        self.assertIn('normal Python raised AssertionError', feedback)
        self.assertIn('python -O returned OUTSIDE_MARKER', feedback)
        self.assertIn('is_relative_to', feedback)
