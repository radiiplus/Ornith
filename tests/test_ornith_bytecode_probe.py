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
        held_out = probe.verify_fixture(probe.DEBUG_FIXTURE, 'ValueError')
        self.assertTrue(held_out['verified'])

    def test_optimized_bytecode_omits_guard_but_reads_file(self):
        source = probe.FIXTURE.read_text()
        bytecode = probe.disassemble_optimized(source)
        self.assertNotIn('is_relative_to', bytecode)
        self.assertIn('read_text', bytecode)
        delta = probe.bytecode_delta(source)
        self.assertEqual(delta['assert_statements_in_source'], 1)
        self.assertIn('is_relative_to', delta['methods_removed_in_optimized'])
        self.assertIn('read_text', delta['methods_remaining_in_optimized'])
        debug_delta = probe.bytecode_delta(probe.DEBUG_FIXTURE.read_text())
        self.assertEqual(debug_delta['assert_statements_in_source'], 0)
        self.assertIn('is_relative_to', debug_delta['unreachable_methods_in_optimized'])

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
        answer = ('{"guard":"absent",'
                  '"normal":"blocked","optimized":"reads_outside",'
                  '"cause":"assert_removed",'
                  '"evidence":"The assertion is omitted in optimized bytecode."}')
        self.assertTrue(probe.assess_reasoned_answer(answer, truth)['passed'])
        self.assertTrue(probe.assess_reasoned_answer(answer, truth)['format_valid'])
        incorrect = answer.replace('"absent"', '"present"')
        self.assertFalse(probe.assess_reasoned_answer(incorrect, truth)['prediction_passed'])
        wrong_cause = answer.replace('"assert_removed"', '"injection"')
        self.assertFalse(probe.assess_reasoned_answer(wrong_cause, truth)['cause_passed'])
        observed = ('{"cause":"is_relative_to assertion was removed during optimization",'
                    '"evidence":"The guard assert report.is_relative_to(base) was stripped in '
                    'optimized bytecode; normal Python raised AssertionError before reading '
                    'while python -O returned OUTSIDE_MARKER from ../outside.txt",'
                    '"confidence":0.9}')
        normalized = probe.assess_reasoned_answer(observed, truth)
        self.assertTrue(normalized['passed'])
        self.assertFalse(normalized['format_valid'])

    def test_feedback_uses_checked_behavior_and_disassembly(self):
        feedback = probe.checker_feedback(
            {'answer': {'cause': 'injection'}}, probe.verify_fixture(),
            probe.bytecode_delta(probe.FIXTURE.read_text()))
        self.assertIn('normal Python raised AssertionError', feedback)
        self.assertIn('python -O returned OUTSIDE_MARKER', feedback)
        self.assertIn('is_relative_to', feedback)

    def test_feedback_stops_after_success_and_retries_once_after_failure(self):
        truth = probe.verify_fixture()
        delta = probe.bytecode_delta(probe.FIXTURE.read_text())
        correct = ('{"guard":"absent","normal":"blocked",'
                   '"optimized":"reads_outside","cause":"assert_removed",'
                   '"evidence":"The assert guard is omitted under python -O."}')
        calls = []

        def succeed(prompt):
            calls.append(prompt)
            return correct, 1.0

        self.assertEqual(len(list(probe.feedback_attempts(succeed, 'source', truth, delta))), 1)
        self.assertEqual(len(calls), 1)
        calls.clear()

        def correct_after_feedback(prompt):
            calls.append(prompt)
            return ('{"guard":"present","normal":"blocked",'
                    '"optimized":"blocked","cause":"injection","evidence":"wrong"}', 1.0) \
                if len(calls) == 1 else (correct, 1.0)

        attempts = list(probe.feedback_attempts(correct_after_feedback, 'source', truth, delta))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(calls), 2)
        self.assertFalse(attempts[0][0]['passed'])
        self.assertTrue(attempts[1][0]['passed'])
        self.assertIn('OUTSIDE_MARKER', calls[1])

    def test_bytecode_delta_does_not_flag_a_runtime_guard(self):
        source = ('from pathlib import Path\n'
                  'def read_report(workspace, name):\n'
                  '    base = Path(workspace).resolve()\n'
                  '    report = (base / name).resolve()\n'
                  '    if not report.is_relative_to(base):\n'
                  '        raise ValueError("outside")\n'
                  '    return report.read_text()\n')
        delta = probe.bytecode_delta(source)
        self.assertNotIn('is_relative_to', delta['methods_removed_in_optimized'])
        self.assertIn('is_relative_to', delta['methods_remaining_in_optimized'])
        self.assertFalse(probe.verified_finding(probe.verify_fixture(), delta)['verified'])

    def test_verified_finding_distinguishes_removed_and_unreachable_guards(self):
        first = probe.verified_finding(probe.verify_fixture(),
                                       probe.bytecode_delta(probe.FIXTURE.read_text()))
        held_out = probe.verified_finding(probe.verify_fixture(probe.DEBUG_FIXTURE, 'ValueError'),
                                          probe.bytecode_delta(probe.DEBUG_FIXTURE.read_text()))
        self.assertEqual(first['cause'], 'assert_removed')
        self.assertEqual(held_out['cause'], 'debug_guard_removed')
        self.assertIn('constant-false jump', held_out['evidence'])
