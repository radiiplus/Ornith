import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    'benchmark_inference', Path(__file__).resolve().parents[1] / 'scripts/benchmark_inference.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class MonitorTests(unittest.TestCase):
    def test_parallel_slots_keep_per_slot_context(self):
        engine = benchmark.LlamaCppAdapter('server', 'ornith.gguf', 2048, 1, 8080,
                                           'mmap', parallel=2)
        self.assertEqual(engine.command[engine.command.index('--parallel') + 1], '2')
        self.assertEqual(engine.command[engine.command.index('--ctx-size') + 1], '2048')
        with self.assertRaises(ValueError):
            benchmark.LlamaCppAdapter('server', 'ornith.gguf', 512, 1, 8080,
                                      'mmap', parallel=2)

    def test_cgroup_pressure_stops_before_slow_process_sampling(self):
        stop_engine = Mock()
        monitor = benchmark.Monitor(123, on_limit=stop_engine, max_cgroup_full_pressure=50)
        with patch.object(benchmark, 'cgroup_full_pressure', return_value=80), \
             patch.object(benchmark, 'counters', side_effect=AssertionError('sampled too late')):
            monitor.run()
        self.assertIn('Cgroup memory full pressure exceeded', monitor.breach_reason)
        stop_engine.assert_called_once_with()
        self.assertEqual(monitor.samples, [])


if __name__ == '__main__':
    unittest.main()
