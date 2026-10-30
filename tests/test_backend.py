import io
import json
import unittest
from urllib.error import HTTPError

from xing_specialist.backend import BackendError, GenerationLimits, Message, OpenAICompatibleBackend


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class BackendTests(unittest.TestCase):
    def test_generation_and_limits(self):
        seen = {}

        def transport(request, timeout):
            seen['body'] = json.loads(request.data)
            seen['timeout'] = timeout
            return Response(b'{"choices":[{"message":{"content":"answer"}}]}')

        backend = OpenAICompatibleBackend('http://127.0.0.1:8080/v1', 'temporary', transport)
        self.assertEqual(backend.generate([Message('user', 'question')], GenerationLimits(256, 32, 3)), 'answer')
        self.assertEqual(seen['body']['max_tokens'], 32)
        self.assertEqual(seen['timeout'], 3)

    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleBackend('https://remote.example/v1', 'model')

    def test_http_failure_is_explicit(self):
        def transport(request, timeout):
            raise HTTPError(request.full_url, 503, 'busy', {}, None)

        backend = OpenAICompatibleBackend('http://localhost:8080/v1', 'temporary', transport)
        with self.assertRaisesRegex(BackendError, 'HTTP 503'):
            backend.generate([Message('user', 'hello')], GenerationLimits())

    def test_context_budget(self):
        backend = OpenAICompatibleBackend('http://localhost:8080/v1', 'temporary')
        with self.assertRaisesRegex(BackendError, 'context budget'):
            backend.generate([Message('user', 'x' * 1000)], GenerationLimits(64, 16))

    def test_stream_separates_reasoning_and_answer_and_reports_tps(self):
        seen = {}
        events = []

        def transport(request, timeout):
            seen['body'] = json.loads(request.data)
            return Response(b''.join([
                b'data: {"choices":[{"delta":{"reasoning_content":"Check the boundary."}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"{\\"answer\\":"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"42}"},"finish_reason":"stop"}]}\n\n',
                b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3},'
                b'"timings":{"predicted_n":3,"predicted_per_second":2.5}}\n\n',
                b'data: [DONE]\n\n',
            ]))

        backend = OpenAICompatibleBackend('http://localhost:8080/v1', 'temporary', transport,
                                          lambda kind, text: events.append((kind, text)),
                                          thinking_budget=48,
                                          response_format={'type': 'json_object'})
        self.assertEqual(backend.generate([Message('user', 'question')],
                                          GenerationLimits(256, 32, 3)), '{"answer":42}')
        self.assertTrue(seen['body']['stream'])
        self.assertTrue(seen['body']['stream_options']['include_usage'])
        self.assertTrue(seen['body']['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(seen['body']['reasoning_budget_tokens'], 48)
        self.assertEqual(seen['body']['response_format']['type'], 'json_object')
        self.assertEqual(events[0], ('reasoning_content', 'Check the boundary.'))
        self.assertEqual(backend.last_metrics['generation_tps'], 2.5)
        self.assertEqual(backend.last_metrics['reasoning_chars'], len('Check the boundary.'))
        self.assertEqual(backend.last_metrics['completion_tokens'], 3)


if __name__ == '__main__':
    unittest.main()
