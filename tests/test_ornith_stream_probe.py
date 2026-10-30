from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from probe_ornith_stream import assess_answer


class StreamProbeTests(unittest.TestCase):
    def test_accepts_case_equivalent_checked_answer(self):
        result = assess_answer('{"answer":"restart the service"}')
        self.assertTrue(result['passed'])
        self.assertTrue(result['format_valid'])

    def test_rejects_wrong_action_and_wrong_key(self):
        self.assertFalse(assess_answer('{"answer":"validate the health check"}')['passed'])
        alias = assess_answer('{"next_action":"restart the service"}')
        self.assertTrue(alias['semantic_passed'])
        self.assertFalse(alias['format_valid'])
        self.assertFalse(alias['passed'])


if __name__ == '__main__':
    unittest.main()
