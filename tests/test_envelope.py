import unittest

from xing_specialist.envelope import GIB, check_values


class SmallModelEnvelopeTests(unittest.TestCase):
    def test_one_core_and_memory_limits(self):
        period = 100000
        check_values(50000, period, 4 * GIB, 6 * GIB, 19 * GIB)
        check_values(100000, period, 4 * GIB, 6 * GIB, 19 * GIB)
        for values in ((150000, period, 4 * GIB, 6 * GIB, 19 * GIB),
                       (100000, period, 5 * GIB, 6 * GIB, 19 * GIB),
                       (100000, period, 4 * GIB, 7 * GIB, 19 * GIB),
                       (100000, period, 4 * GIB, 6 * GIB, 20 * GIB)):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                check_values(*values)


if __name__ == '__main__':
    unittest.main()
