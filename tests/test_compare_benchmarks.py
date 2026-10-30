import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'compare_benchmarks', Path(__file__).resolve().parents[1] / 'scripts/compare_benchmarks.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def artifact(self, status='passed'):
        return {
            'model_name': 'sample',
            'status': status,
            'context_tokens': 512,
            'generation_limit_tokens': 8,
            'prompt_sha256': 'prompt-hash',
            'runs': [{'tokens_per_second_after_first_token': 3.0}],
        }

    def test_failed_model_is_not_compared_as_a_speed_result(self):
        result = comparison.compare(self.artifact('failed'), self.artifact())
        self.assertFalse(result['comparable'])
        self.assertIn('Both models must complete inference', result['issues'][0])

    def test_raw_samples_correct_stale_peak_fields(self):
        artifact = self.artifact()
        artifact.update({
            'peak_rss_bytes': 1024,
            'major_faults_delta': 1,
            'samples': [
                {'rss_high_water_bytes': 1024, 'major_faults': 2},
                {'rss_high_water_bytes': 2 * 1024 ** 3, 'major_faults': 20},
            ],
        })
        summary = comparison.summarize(artifact)
        self.assertEqual(summary['peak_rss_gib'], 2.0)
        self.assertEqual(summary['major_faults'], 18)

    def test_matching_completed_runs_can_be_compared(self):
        result = comparison.compare(self.artifact(), self.artifact())
        self.assertTrue(result['comparable'])
        self.assertEqual(result['issues'], [])


if __name__ == '__main__':
    unittest.main()
