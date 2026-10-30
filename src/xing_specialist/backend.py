"""Model-independent generation contract and local HTTP implementation."""

from dataclasses import dataclass
import json
import time
from typing import Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationLimits:
    context_tokens: int = 2048
    output_tokens: int = 512
    timeout_seconds: float = 120.0

    def __post_init__(self):
        if min(self.context_tokens, self.output_tokens, self.timeout_seconds) <= 0:
            raise ValueError('Generation limits must be positive')
        if self.output_tokens >= self.context_tokens:
            raise ValueError('Output limit must be smaller than context limit')


class ModelBackend(Protocol):
    def generate(self, messages: Sequence[Message], limits: GenerationLimits) -> str:
        ...


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, model: str, transport: Callable = urlopen,
                 on_chunk: Callable = None, thinking_budget: int = 0,
                 response_format: dict = None):
        parsed = urlparse(base_url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('Initial backend must use a local HTTP endpoint')
        if not model:
            raise ValueError('Model name is required')
        if not 0 <= thinking_budget <= 128:
            raise ValueError('Thinking budget must be between 0 and 128 tokens')
        self.endpoint = base_url.rstrip('/') + '/chat/completions'
        self.model = model
        self.transport = transport
        self.on_chunk = on_chunk
        self.thinking_budget = thinking_budget
        self.response_format = response_format
        self.last_metrics = {}

    def _read_stream(self, response):
        content = []
        usage = timings = {}
        first_chunk_at = None
        reasoning_chars = 0
        finish_reason = None
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            payload = raw[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            usage = event.get('usage') or usage
            timings = event.get('timings') or timings
            for choice in event.get('choices', []):
                finish_reason = choice.get('finish_reason') or finish_reason
                delta = choice.get('delta') or {}
                for kind in ('reasoning_content', 'content'):
                    part = delta.get(kind)
                    if isinstance(part, str) and part:
                        if first_chunk_at is None:
                            first_chunk_at = time.monotonic()
                        if kind == 'content':
                            content.append(part)
                        else:
                            reasoning_chars += len(part)
                        self.on_chunk(kind, part)
        return ''.join(content), usage, timings, first_chunk_at, reasoning_chars, finish_reason

    def generate(self, messages: Sequence[Message], limits: GenerationLimits) -> str:
        if not messages or any(m.role not in ('system', 'user', 'assistant') for m in messages):
            raise ValueError('Messages must have supported roles')
        # This conservative character bound is only a preflight; the server enforces its tokenizer limit.
        if sum(len(m.content) for m in messages) > (limits.context_tokens - limits.output_tokens) * 3:
            raise BackendError('Prompt exceeds conservative context budget')
        request_body = {
            'model': self.model,
            'messages': [m.__dict__ for m in messages],
            'max_tokens': limits.output_tokens,
            'temperature': 0,
            'stream': self.on_chunk is not None,
        }
        if self.on_chunk is not None:
            request_body['stream_options'] = {'include_usage': True}
        if self.thinking_budget:
            request_body['chat_template_kwargs'] = {'enable_thinking': True}
            request_body['reasoning_format'] = 'deepseek'
            request_body['reasoning_budget_tokens'] = self.thinking_budget
        if self.response_format is not None:
            request_body['response_format'] = self.response_format
        request = Request(self.endpoint, data=json.dumps(request_body).encode(),
                          headers={'Content-Type': 'application/json'})
        started = time.monotonic()
        self.last_metrics = {}
        try:
            with self.transport(request, timeout=limits.timeout_seconds) as response:
                if self.on_chunk is None:
                    payload = json.load(response)
                    text = payload['choices'][0]['message']['content']
                    usage = payload.get('usage') or {}
                    timings = payload.get('timings') or {}
                    first_chunk_at = None
                    reasoning_chars = len(payload['choices'][0]['message'].get('reasoning_content') or '')
                    finish_reason = payload['choices'][0].get('finish_reason')
                else:
                    text, usage, timings, first_chunk_at, reasoning_chars, finish_reason = (
                        self._read_stream(response))
        except HTTPError as exc:
            raise BackendError(f'Model server returned HTTP {exc.code}') from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise BackendError(f'Model server unavailable or timed out: {type(exc).__name__}') from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BackendError('Model server returned invalid JSON') from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError('Model server response lacks assistant content') from exc
        if not isinstance(text, str) or not text.strip():
            raise BackendError('Model server returned empty assistant content')
        elapsed = time.monotonic() - started
        completion_tokens = usage.get('completion_tokens') or timings.get('predicted_n')
        generation_tps = timings.get('predicted_per_second')
        if not generation_tps and completion_tokens and first_chunk_at is not None:
            decode_seconds = time.monotonic() - first_chunk_at
            if decode_seconds > 0 and completion_tokens > 1:
                generation_tps = (completion_tokens - 1) / decode_seconds
        self.last_metrics = {
            'request_seconds': round(elapsed, 2),
            'time_to_first_chunk_seconds': (round(first_chunk_at - started, 2)
                                            if first_chunk_at is not None else None),
            'prompt_tokens': usage.get('prompt_tokens') or timings.get('prompt_n'),
            'completion_tokens': completion_tokens,
            'generation_tps': round(generation_tps, 3) if generation_tps else None,
            'reasoning_chars': reasoning_chars,
            'finish_reason': finish_reason,
        }
        return text
