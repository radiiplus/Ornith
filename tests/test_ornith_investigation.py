from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_ornith_investigation as probe


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.executor = probe.IsolatedExecutor(Path(self.folder.name), timeout_seconds=10,
                                                memory_mib=256)

    def test_both_regressions_fail_and_oracle_repairs_pass(self):
        for case in probe.CASES.values():
            with self.subTest(case=case['target']):
                checked = probe.verify_fixture(case, self.executor)
                self.assertNotEqual(checked['baseline']['exit_code'], 0)
                self.assertEqual(checked['oracle']['exit_code'], 0)

    def test_fixture_signature_ignores_unittest_elapsed_time(self):
        fixture = probe.verify_fixture(probe.CASES['tenant_reports'], self.executor)
        other = probe.json.loads(probe.json.dumps(fixture))
        other['oracle']['output'] = probe.re.sub(r'in [\d.]+s', 'in 0.371s',
                                                other['oracle']['output'])
        self.assertEqual(probe.fixture_signature(fixture), probe.fixture_signature(other))

    def test_final_requires_inspection_and_passing_edit(self):
        case = probe.CASES['tenant_reports']
        investigation = probe.Investigation(case, self.executor)
        proposed = {'target': case['target'], 'old': case['oracle_old'],
                    'new': case['oracle_new']}
        self.assertFalse(investigation.final_grade(proposed)['passed'])
        self.assertIn('Unknown path', investigation.observe(
            {'action': 'read', 'target': '../tenant_reports/reporting/service.py'}))
        self.assertEqual(investigation.inspected, set())
        investigation.observe({'action': 'read', 'target': case['target']})
        self.assertTrue(investigation.final_grade(proposed)['passed'])
        proposed['new'] = "key = user['id']"
        self.assertFalse(investigation.final_grade(proposed)['passed'])

    def test_multiline_wrong_file_edit_gets_test_feedback(self):
        case = probe.CASES['tenant_reports']
        investigation = probe.Investigation(case, self.executor)
        investigation.observe({'action': 'read', 'target': case['decoy']})
        grade = investigation.final_grade({
            'target': case['decoy'],
            'old': '    def get(self, key):\n        return self.entries.get(key)',
            'new': '    def get(self, key, tenant):\n        return self.entries.get((key, tenant))'})
        self.assertFalse(grade['passed'])
        self.assertFalse(grade['target_correct'])
        self.assertIn('TypeError', grade['test']['output'])

    def test_trajectory_metrics_distinguish_exploration_from_result(self):
        case = probe.CASES['discount_shipping']
        investigation = probe.Investigation(case, self.executor)
        for target in (case['decoy'], case['target']):
            observation = investigation.observe({'action': 'read', 'target': target})
            investigation.steps.append({'proposal': {'action': 'read', 'target': target,
                                                     'why': target, 'expect': 'Inspect fee input'},
                                        'observation': observation})
        metrics = investigation.metrics()
        self.assertTrue(metrics['true_file_inspected'])
        self.assertTrue(metrics['moved_from_decoy_to_cause'])
        self.assertEqual(metrics['predictions_recorded'], 2)
        self.assertEqual(metrics['distinct_files_inspected'], 2)

    def test_loop_carries_observation_and_checks_final_edit(self):
        case = probe.CASES['discount_shipping']
        prompts = []
        responses = [
            {'action': 'read', 'target': case['target'], 'why': 'Order may pass the wrong amount',
             'expect': 'See which amount is sent to fee', 'old': '', 'new': ''},
            {'action': 'final', 'target': case['target'], 'why': 'fee gets original subtotal',
             'expect': 'Discounted subtotal will trigger delivery',
             'old': case['oracle_old'], 'new': case['oracle_new']},
        ]

        def fake_ask(_port, prompt):
            prompts.append(prompt)
            return probe.json.dumps(responses.pop(0)), 0.01, {}, None

        output = Path(self.folder.name) / 'result.json'
        result = {'fixture': probe.verify_fixture(case, self.executor), 'steps': []}
        with patch.object(probe, 'ask', side_effect=fake_ask):
            probe.run_case(case, self.executor, 0, 3, output, result)
        self.assertIn('delivery = fee(subtotal)', prompts[1])
        self.assertTrue(result['grade']['passed'])
        self.assertTrue(result['metrics']['true_file_inspected'])
        self.assertEqual(len(result['steps']), 2)

    def test_failed_edit_feedback_allows_recovery(self):
        case = probe.CASES['discount_shipping']
        responses = [
            {'action': 'read', 'target': case['target'], 'why': 'Inspect order',
             'expect': 'Find shipping input', 'old': '', 'new': ''},
            {'action': 'final', 'target': case['target'], 'why': 'Try subtotal',
             'expect': 'Tests should pass', 'old': case['oracle_old'],
             'new': 'delivery = fee(subtotal + 1)'},
            {'action': 'final', 'target': case['target'], 'why': 'Use discounted amount',
             'expect': 'Tests should pass', 'old': case['oracle_old'],
             'new': case['oracle_new']},
        ]
        prompts = []

        def fake_ask(_port, prompt):
            prompts.append(prompt)
            return probe.json.dumps(responses.pop(0)), 0.01, {}, None

        result = {'fixture': probe.verify_fixture(case, self.executor), 'steps': []}
        with patch.object(probe, 'ask', side_effect=fake_ask):
            probe.run_case(case, self.executor, 0, 3,
                           Path(self.folder.name) / 'retry.json', result)
        self.assertIn('"passed": false', prompts[2])
        self.assertNotIn('target_correct', prompts[2])
        self.assertEqual(result['metrics']['final_attempts'], 2)
        self.assertTrue(result['metrics']['recovered_after_failed_edit'])
        self.assertTrue(result['grade']['passed'])

    def test_resume_reuses_inspection_without_repeating_action(self):
        case = probe.CASES['tenant_reports']
        result = {'fixture': probe.verify_fixture(case, self.executor), 'steps': [
            {'number': 1,
             'proposal': {'action': 'read', 'target': case['target'],
                          'why': 'Inspect cache key', 'expect': 'See tenant in key'},
             'observation': (case['root'] / case['target']).read_text(),
             'seconds': 1.0}]}
        final = {'action': 'final', 'target': case['target'],
                 'why': 'Tenant is omitted from key', 'expect': 'Tests pass',
                 'old': case['oracle_old'], 'new': case['oracle_new']}
        prompts = []

        def fake_ask(_port, prompt):
            prompts.append(prompt)
            return probe.json.dumps(final), 0.01, {}, None

        with patch.object(probe, 'ask', side_effect=fake_ask):
            probe.run_case(case, self.executor, 0, 2,
                           Path(self.folder.name) / 'resume.json', result)
        self.assertIn('Tenant is omitted', result['steps'][-1]['proposal']['why'])
        self.assertIn('key = user', prompts[0])
        self.assertTrue(result['grade']['passed'])
        self.assertEqual(len(result['steps']), 2)


if __name__ == '__main__':
    unittest.main()
