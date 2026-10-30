import unittest

from xing_specialist.planner import Planner


class PlannerTests(unittest.TestCase):
    def test_dependencies_and_verifiable_steps(self):
        steps = Planner().plan('Fix parser')
        self.assertEqual([step.key for step in steps], ['inspect', 'hypothesize', 'change', 'verify', 'record'])
        self.assertFalse(Planner.ready(steps[2], {'inspect'}))
        self.assertTrue(Planner.ready(steps[2], {'inspect', 'hypothesize'}))
        self.assertTrue(all(step.expected for step in steps))

    def test_failed_action_needs_new_evidence_and_budget(self):
        planner = Planner(max_retries=2)
        self.assertTrue(planner.may_retry('patch A', 'failure 1'))
        planner.failed('patch A', 'failure 1')
        self.assertFalse(planner.may_retry('patch A', 'failure 1'))
        self.assertTrue(planner.may_retry('patch A', 'failure 2'))
        planner.failed('patch A', 'failure 2')
        self.assertFalse(planner.may_retry('patch A', 'failure 3'))


if __name__ == '__main__':
    unittest.main()
