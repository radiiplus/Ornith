from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from xing_specialist.agent import EngineeringAgent
from xing_specialist.backend import GenerationLimits, Message
from xing_specialist.dual_router import (DualModelRouter, PlanProgress, RoleBackend,
                                         objective_gate, validate_actions)
from xing_specialist.memory import MemoryStore


GATE = {'objective': 'Fix add so the test passes', 'subgoals': ['Fix add'],
        'assumptions': [], 'done_when': 'The test passes'}
STEP = {'id': 'answer', 'objective': 'Choose an answer', 'done_when': 'Checker accepts it',
        'check': {'kind': 'checked_answer', 'pattern': ''}}


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.roles = []
        self.prompts = []
        self.schemas = []
        self.message_roles = []

    def generate(self, role, messages, _limits, _schema=None):
        self.roles.append(role)
        self.prompts.append(messages[-1].content)
        self.schemas.append(_schema)
        self.message_roles.append([message.role for message in messages])
        return json.dumps(self.responses.pop(0)), {'request_seconds': 0.01, 'generation_tps': 10.0}


class DualRouterTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        (self.root / 'calc.py').write_text('def add(a, b):\n    return a - b\n')
        self.store = MemoryStore(self.root / 'memory.sqlite3')
        self.addCleanup(self.store.close)
        self.command = (('python3', '-m', 'unittest', 'discover', '-s', 'tests'),)

    def router(self, responses, **kwargs):
        provider = FakeProvider(responses)
        revisions = kwargs.pop('max_revisions', 1)
        router = DualModelRouter(provider, self.store, object(), max_attempts=1,
                                 max_revisions=revisions, **kwargs)
        return router, provider

    def test_gate_rejects_drift_and_unapproved_actions(self):
        with self.assertRaises(ValueError):
            objective_gate('Fix adult age boundary', {
                'objective': 'Summarize deployment', 'subgoals': [],
                'assumptions': [], 'done_when': 'Deployment summary'})
        with self.assertRaisesRegex(ValueError, 'negative constraint'):
            objective_gate('Do not deploy the patch before tests pass', {
                'objective': 'Deploy the patch before tests pass', 'subgoals': [],
                'assumptions': [], 'done_when': 'Patch deployed after tests'})
        with self.assertRaises(ValueError):
            objective_gate('Passing tests is sufficient for deployment. A build was deployed. '
                           'Must it have passed tests?', {
                               'objective': 'Pass the tests for the application',
                               'subgoals': ['Must it have passed the tests?'],
                               'assumptions': [], 'done_when': 'After deployment is complete'})
        with self.assertRaisesRegex(ValueError, 'unstated assumptions'):
            objective_gate('Fix add so the test passes', dict(
                GATE, assumptions=['Inputs are strings']))
        with self.assertRaisesRegex(ValueError, 'Definition of done'):
            objective_gate('Fix add so the test passes', dict(
                GATE, done_when='After 24 hours of coding'))
        with self.assertRaisesRegex(ValueError, 'passing tests'):
            objective_gate('Fix add so the test passes', dict(
                GATE, done_when='After add completes successfully'))
        with self.assertRaises(ValueError):
            validate_actions({'actions': [{'kind': 'repair', 'path': 'other.py', 'answer': ''}],
                              'parked_observations': []}, ('calc.py',))

    def test_gate_accepts_question_restatement_without_answer_verb(self):
        question = ('A implies B, B implies C, C implies D, and D implies E. '
                    'Either A or F is true, and F is false. Must E be true?')
        restatement = {'objective': question, 'subgoals': ['Determine whether E follows'],
                       'assumptions': [], 'done_when': 'E is checked as true or false'}
        self.assertEqual(objective_gate(question, restatement), restatement)

    def test_repair_schema_limits_edit_paths(self):
        provider = FakeProvider([{'hypothesis': 'wrong operator', 'expected': 'sum passes',
                                  'edits': [{'path': 'calc.py', 'old': 'a - b',
                                             'new': 'a + b'}]}])
        RoleBackend(provider, 'execution', ('calc.py',)).generate(
            (Message('user', 'Fix add'),), GenerationLimits())
        path = provider.schemas[0]['properties']['edits']['items']['properties']['path']
        self.assertEqual(path['enum'], ['calc.py'])

    def test_checked_answer_and_parked_recall(self):
        prior = self.store.task('Earlier work')
        self.store.park_observation(prior, 'Tenant cache omits tenant key',
                                    'review of service', 'tenant cache', 'Might leak reports')
        self.assertEqual(len(self.store.recall_parked('Fix tenant cache isolation')), 1)
        self.assertEqual(self.store.recall_parked('Calculate pump rates'), [])
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': '42'}],
             'parked_observations': []}], max_revisions=0)
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda value: None if value == '42' else
                             (_ for _ in ()).throw(ValueError('Wrong')))
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(provider.roles, ['reasoning', 'reasoning', 'execution'])
        self.assertEqual(len(outcome.steps), 1)

    def test_exact_answer_uses_one_step_and_one_answer_action(self):
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'ok'}],
             'parked_observations': []}], max_revisions=0)
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(provider.roles.count('execution'), 1)
        self.assertEqual(provider.schemas[1]['properties']['steps']['maxItems'], 1)
        self.assertEqual(provider.schemas[1]['properties']['steps']['items']['properties']
                         ['check']['properties']['kind']['enum'], ['checked_answer'])
        self.assertEqual(provider.schemas[2]['properties']['actions']['items']
                         ['properties']['path']['enum'], ['', 'calc.py'])
        self.assertEqual(provider.schemas[2]['properties']['actions']['maxItems'], 1)
        self.assertEqual(provider.schemas[2]['properties']['actions']['items']
                         ['properties']['kind']['enum'], ['answer'])

    def test_malformed_calculator_expression_gets_a_bounded_retry(self):
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'The result is 2'}],
             'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': '1+1'}],
             'parked_observations': []}], max_revisions=0)
        router.max_attempts = 2

        def check(answer):
            if answer != '1+1':
                raise SyntaxError('Expression required')

        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, check, answer_format='arithmetic')
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(provider.roles.count('execution'), 2)
        self.assertIn('Arithmetic answer must be an expression without prose',
                      provider.prompts[3])

    def test_arithmetic_handoff_rejects_prose_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'without prose'):
            validate_actions({'actions': [{'kind': 'answer', 'path': '',
                                           'answer': 'The answer is 2'}],
                              'parked_observations': []}, ('calc.py',),
                             answer_format='arithmetic')

    def test_user_only_prompt_style_keeps_instructions_in_user_message(self):
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'ok'}],
             'parked_observations': []}], max_revisions=0,
            prompt_style='user-only')
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(provider.message_roles, [['user']] * 3)
        self.assertIn('Return one JSON object only', provider.prompts[0])

    def test_role_aware_prompt_style_keeps_vibe_system_message(self):
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'ok'}],
             'parked_observations': []}], max_revisions=0,
            prompt_style='role-aware')
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(provider.message_roles, [['system', 'user'],
                                                  ['system', 'user'], ['user']])

    def test_verified_logic_context_reaches_both_roles_and_limits_answer(self):
        task_context = {'variables': ['a', 'b'], 'premises': ['not a or b', 'a'],
                        'conclusion': 'b'}
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'entailed'}],
             'parked_observations': []}], max_revisions=0)
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _value: None,
                             answer_format='logic', task_context=task_context)
        self.assertEqual(outcome.status, 'verified')
        self.assertIn('not a or b', provider.prompts[1])
        self.assertIn('not a or b', provider.prompts[2])
        answer_schema = provider.schemas[2]['properties']['actions']['items']['properties']['answer']
        self.assertEqual(answer_schema['enum'], ['entailed', 'not_entailed'])

    def test_plan_progress_uses_available_actions_and_final_checker(self):
        task_context = {'initial': ['ready'], 'goal': ['done'],
                        'actions': {'validate': {'requires': ['ready'], 'adds': ['valid'],
                                                 'removes': []},
                                    'finish': {'requires': ['valid'], 'adds': ['done'],
                                               'removes': []}}}
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'advance', 'path': '', 'answer': 'validate'}],
             'parked_observations': []},
            {'actions': [{'kind': 'advance', 'path': '', 'answer': 'finish'}],
             'parked_observations': []}], max_revisions=0)
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda value: None if value == ['validate', 'finish']
                             else (_ for _ in ()).throw(ValueError('Wrong')),
                             answer_format='plan', task_context=task_context)
        self.assertEqual(outcome.status, 'verified')
        first_schema = provider.schemas[2]['properties']['actions']['items']['properties']['answer']
        second_schema = provider.schemas[3]['properties']['actions']['items']['properties']['answer']
        self.assertEqual(first_schema['enum'], ['validate'])
        self.assertEqual(second_schema['enum'], ['finish'])
        self.assertIn('Actions already applied: ["validate"]', provider.prompts[3])
        with self.assertRaisesRegex(ValueError, 'not currently available'):
            validate_actions({'actions': [{'kind': 'advance', 'path': '',
                                           'answer': 'finish'}], 'parked_observations': []},
                             ('calc.py',), answer_format='plan',
                             available_actions=('validate',))

    def test_plan_progress_prefers_actions_that_preserve_achieved_goals(self):
        progress = PlanProgress({
            'initial': ['ready'], 'goal': ['partial', 'done'],
            'actions': {
                'start': {'requires': ['ready'], 'adds': ['partial'], 'removes': []},
                'finish': {'requires': ['partial'], 'adds': ['done'], 'removes': []},
                'undo': {'requires': ['partial'], 'adds': [], 'removes': ['partial']},
            }})
        progress.advance('start')
        self.assertEqual(progress.available(), ['finish'])

    def test_repair_step_requires_progress_after_inspection(self):
        repair_step = dict(STEP, check={'kind': 'tests_pass', 'pattern': ''})
        router, provider = self.router([
            GATE, {'steps': [repair_step], 'parked_observations': []},
            {'actions': [{'kind': 'inspect', 'path': 'calc.py', 'answer': ''}],
             'parked_observations': []},
            {'actions': [{'kind': 'repair', 'path': 'calc.py', 'answer': ''}],
             'parked_observations': []}], max_revisions=0)
        staged = SimpleNamespace(task_id=2, status='needs_review',
                                 detail='Patch tested in staging only')
        with patch.object(EngineeringAgent, 'run', return_value=staged):
            outcome = router.run('Fix add so the test passes', self.root,
                                 ('calc.py',), self.command)
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(provider.schemas[3]['properties']['actions']['items']
                         ['properties']['kind']['enum'], ['repair'])

    def test_only_source_grounded_notes_are_parked(self):
        note = {'observation': 'calc.py contains return a - b',
                'context': 'Inspection of calc.py', 'tag': 'subtraction',
                'reason': 'May matter for another repair'}
        ungrounded = dict(note, observation='An unrelated clock was wrong')
        evidence_step = {'id': 'inspect', 'objective': 'Find source calculation',
                         'done_when': 'The source calculation is found',
                         'check': {'kind': 'evidence_found', 'pattern': 'return a - b'}}
        repair_step = dict(STEP, check={'kind': 'tests_pass', 'pattern': ''})
        router, _ = self.router([
            GATE, {'steps': [evidence_step, repair_step], 'parked_observations': [ungrounded]},
            {'actions': [{'kind': 'inspect', 'path': 'calc.py', 'answer': ''}],
             'parked_observations': [note]},
            {'actions': [{'kind': 'repair', 'path': 'calc.py', 'answer': ''}],
             'parked_observations': []}], max_revisions=0)
        staged = SimpleNamespace(task_id=2, status='needs_review',
                                 detail='Patch tested in staging only')
        with patch.object(EngineeringAgent, 'run', return_value=staged):
            outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                                 self.command)
        self.assertEqual(outcome.status, 'verified')
        parked = self.store.db.execute('SELECT observation FROM parked_observations').fetchall()
        self.assertEqual([row[0] for row in parked], [note['observation']])

    def test_failed_step_revises_plan_with_evidence(self):
        revision = {'broken_assumption': 'First answer was correct',
                    'change': 'Try a corrected answer', 'steps': [STEP],
                    'parked_observations': []}
        router, _ = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'wrong'}],
             'parked_observations': []}, revision,
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'right'}],
             'parked_observations': []}])

        def check(value):
            if value != 'right':
                raise ValueError('Not correct')

        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, check)
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(outcome.plan_revisions, 1)
        logged = self.store.db.execute('''SELECT detail FROM actions WHERE kind="plan_revision"
                                          ORDER BY id DESC LIMIT 1''').fetchone()
        self.assertIn('First answer was correct', logged[0])
        self.assertIn('Not correct', logged[0])

    def test_invalid_revision_leaves_failed_step_for_review(self):
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'wrong'}],
             'parked_observations': []}, 'truncated revision'])
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _answer: (_ for _ in ()).throw(
                                 ValueError('Wrong')))
        self.assertEqual(outcome.status, 'needs_review')
        self.assertEqual(outcome.plan_revisions, 0)
        self.assertEqual(provider.schemas[-1]['properties']['steps']['maxItems'], 1)
        kinds = [row[0] for row in self.store.db.execute('SELECT kind FROM actions')]
        self.assertIn('plan_revision_rejected', kinds)

    def test_missing_file_evidence_revises_without_repeated_inspection(self):
        inspect_step = {'id': 'inspect', 'objective': 'Find cache key in file',
                        'done_when': 'Key is present',
                        'check': {'kind': 'evidence_found', 'pattern': 'tenant_key'}}
        repair_step = dict(STEP, check={'kind': 'tests_pass', 'pattern': ''})
        revision = {'broken_assumption': 'The inspected file contained tenant_key',
                    'change': 'Repair the observed source instead', 'steps': [repair_step],
                    'parked_observations': []}
        router, provider = self.router([
            GATE, {'steps': [inspect_step, repair_step], 'parked_observations': []},
            {'actions': [{'kind': 'inspect', 'path': 'calc.py', 'answer': ''}],
             'parked_observations': []}, revision,
            {'actions': [{'kind': 'repair', 'path': 'calc.py', 'answer': ''}],
             'parked_observations': []}])
        staged = SimpleNamespace(task_id=2, status='needs_review',
                                 detail='Patch tested in staging only')
        with patch.object(EngineeringAgent, 'run', return_value=staged):
            outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                                 self.command)
        self.assertEqual(outcome.status, 'verified')
        self.assertEqual(outcome.plan_revisions, 1)
        self.assertEqual(provider.roles.count('execution'), 2)
        logged = self.store.db.execute('''SELECT detail FROM actions WHERE kind="step_outcome"
                                          ORDER BY id LIMIT 1''').fetchone()
        self.assertIn('"contradicted": true', logged[0])

    def test_gate_failure_stops_before_execution(self):
        bad = {'objective': 'Write a poem', 'subgoals': [], 'assumptions': [],
               'done_when': 'Poem done'}
        router, provider = self.router([bad, bad])
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',), self.command)
        self.assertEqual(outcome.status, 'failed')
        self.assertEqual(provider.roles, ['reasoning', 'reasoning'])
        self.assertIn('Restatement omitted key terms', provider.prompts[1])

    def test_known_corpus_is_logged_before_planning(self):
        area = self.root / 'areas_tenant.txt'
        area.write_text('tenant cache key can leak reports across tenants\n')
        gate = {'objective': 'Fix tenant cache report isolation',
                'subgoals': ['Check tenant cache'], 'assumptions': [],
                'done_when': 'Report isolation is fixed'}
        router, provider = self.router([
            gate, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'ok'}],
             'parked_observations': []}], corpus_paths=(area,), max_revisions=0)
        outcome = router.run('Fix tenant cache report isolation', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'verified')
        self.assertIn('areas_tenant.txt', provider.prompts[1])
        logged = self.store.db.execute('''SELECT stdout FROM actions
                                          WHERE kind="known_corpus_check"''').fetchone()
        self.assertIn('tenant cache key', logged[0])

    def test_generic_test_words_do_not_match_unrelated_corpus(self):
        area = self.root / 'areas_checkout.txt'
        area.write_text('The shipping test passes after a change to the fee\n')
        router, provider = self.router([
            GATE, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'ok'}],
             'parked_observations': []}], corpus_paths=(area,), max_revisions=0)
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'verified')
        self.assertNotIn('areas_checkout.txt', provider.prompts[1])

    def test_common_words_do_not_match_corpus(self):
        area = self.root / 'areas_checkout.txt'
        area.write_text('threshold uses a subtotal while tax and final payment use a fee\n')
        gate = {'objective': 'Calculate what pumps do while the leak drains the tank',
                'subgoals': ['Find the final value'], 'assumptions': [],
                'done_when': 'The final tank value is calculated'}
        router, provider = self.router([
            gate, {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'ok'}],
             'parked_observations': []}], corpus_paths=(area,), max_revisions=0)
        outcome = router.run('Calculate what pumps do while the leak drains the tank, '
                             'and find the final value', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'verified')
        self.assertNotIn('areas_checkout.txt', provider.prompts[1])

    def test_reasoning_plan_must_end_with_answer_checker(self):
        test_step = dict(STEP, check={'kind': 'tests_pass', 'pattern': ''})
        router, _ = self.router([
            GATE, {'steps': [test_step], 'parked_observations': []},
            {'steps': [test_step], 'parked_observations': []}], max_revisions=0)
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'failed')
        self.assertEqual(len(outcome.steps), 0)

    def test_plan_correction_uses_checker_feedback(self):
        wrong = dict(STEP, check={'kind': 'tests_pass', 'pattern': ''})
        router, provider = self.router([
            GATE, {'steps': [wrong], 'parked_observations': []},
            {'steps': [STEP], 'parked_observations': []},
            {'actions': [{'kind': 'answer', 'path': '', 'answer': 'ok'}],
             'parked_observations': []}], max_revisions=0)
        outcome = router.run('Fix add so the test passes', self.root, ('calc.py',),
                             self.command, lambda _value: None)
        self.assertEqual(outcome.status, 'verified')
        self.assertIn('Final plan step must use checked_answer', provider.prompts[2])


if __name__ == '__main__':
    unittest.main()
