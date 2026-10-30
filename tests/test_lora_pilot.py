import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from pilot_lora_reasoning import encode_sample, training_rows  # noqa: E402


class TokenizerStub:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self_value = [1, 2, 3]
        return self_value if add_generation_prompt else self_value + [4, 5]


class LoraPilotTests(unittest.TestCase):
    def test_training_labels_mask_the_prompt(self):
        encoded = encode_sample(TokenizerStub(), {'question': 'Q', 'answer': 'A'}, 64)
        self.assertEqual(encoded, ([1, 2, 3, 4, 5], [-100, -100, -100, 4, 5]))

    def test_training_rows_require_question_and_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.jsonl'
            path.write_text(json.dumps({'question': 'Q', 'answer': 'A'}) + '\n')
            self.assertEqual(list(training_rows(path)),
                             [(1, {'question': 'Q', 'answer': 'A'})])
            path.write_text(json.dumps({'question': 'Q', 'answer': 'A',
                                        'verifier': 'secret'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'Invalid GSM8K'):
                list(training_rows(path))


if __name__ == '__main__':
    unittest.main()
