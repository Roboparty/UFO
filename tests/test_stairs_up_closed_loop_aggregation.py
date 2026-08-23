import statistics
import unittest

from humanoidverse.aggregate_stairs_up_closed_loop import _bootstrap_mean_ci


class ClosedLoopAggregationTest(unittest.TestCase):
    def test_bootstrap_constant_values_have_zero_width_interval(self):
        low, high = _bootstrap_mean_ci([0.75] * 10, samples=100, seed=1)
        self.assertEqual(low, 0.75)
        self.assertEqual(high, 0.75)

    def test_bootstrap_is_deterministic_and_bounded(self):
        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        first = _bootstrap_mean_ci(values, samples=1000, seed=7)
        second = _bootstrap_mean_ci(values, samples=1000, seed=7)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 0.0)
        self.assertLessEqual(first[1], 1.0)
        self.assertLess(first[0], statistics.fmean(values))
        self.assertGreater(first[1], statistics.fmean(values))


if __name__ == "__main__":
    unittest.main()
