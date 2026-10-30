#!/usr/bin/env python3
"""Benchmark a local OpenAI-compatible inference engine with host counters."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Protocol
import urllib.error
import urllib.request


GIB = 1024 ** 3
PROMPT = (
    "Write a Python function that parses nested parentheses and returns the "
    "maximum nesting depth. Include validation for unmatched parentheses "
    "and explain its time complexity."
)


def fields(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            result[key] = value.strip()
    return result


def kib(value):
    return int(value.split()[0])


def cgroup_full_pressure(pid):
    try:
        for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines():
            hierarchy, _, group = line.partition('::')
            if hierarchy != '0':
                continue
            pressure_path = Path('/sys/fs/cgroup') / group.lstrip('/') / 'memory.pressure'
            for pressure_line in pressure_path.read_text().splitlines():
                if pressure_line.startswith('full '):
                    return float(dict(item.split('=', 1) for item in pressure_line.split()[1:])['avg10'])
    except (OSError, KeyError, ValueError):
        pass
    return None


def counters(pid):
    status = fields(f'/proc/{pid}/status')
    stat = Path(f'/proc/{pid}/stat').read_text()
    stat_fields = stat[stat.rfind(')') + 2:].split()
    io = fields(f'/proc/{pid}/io')
    memory = fields('/proc/meminfo')
    rollup = fields(f'/proc/{pid}/smaps_rollup')
    return {
        'rss_bytes': kib(status.get('VmRSS', '0 kB')) * 1024,
        'rss_high_water_bytes': kib(status.get('VmHWM', '0 kB')) * 1024,
        'virtual_bytes': kib(status.get('VmSize', '0 kB')) * 1024,
        'process_swap_bytes': kib(status.get('VmSwap', '0 kB')) * 1024,
        'anonymous_bytes': kib(rollup.get('Anonymous', '0 kB')) * 1024,
        'file_pss_bytes': kib(rollup.get('Pss_File', '0 kB')) * 1024,
        'anonymous_pss_bytes': kib(rollup.get('Pss_Anon', '0 kB')) * 1024,
        'major_faults': int(stat_fields[9]),
        'disk_read_bytes': int(io.get('read_bytes', 0)),
        'system_swap_used_bytes': (kib(memory['SwapTotal']) - kib(memory['SwapFree'])) * 1024,
        'mem_available_bytes': kib(memory['MemAvailable']) * 1024,
        'memory_psi': Path('/proc/pressure/memory').read_text().strip().splitlines()
        if Path('/proc/pressure/memory').exists() else None,
    }


def model_mapping(pid, model):
    path = str(model.resolve())
    mapped_bytes = 0
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        if line.endswith(path):
            start, end = line.split()[0].split('-')
            mapped_bytes += int(end, 16) - int(start, 16)
    resident_bytes = 0
    in_model = False
    for line in Path(f'/proc/{pid}/smaps').read_text().splitlines():
        if '-' in line.split()[0] and line.split()[0].count('-') == 1:
            address = line.split()[0]
            in_model = all(c in '0123456789abcdef-' for c in address) and line.rstrip().endswith(path)
        elif in_model and line.startswith('Rss:'):
            resident_bytes += kib(line.split(':', 1)[1]) * 1024
    return {'mapped_virtual_bytes': mapped_bytes, 'resident_bytes': resident_bytes}


class Monitor:
    def __init__(self, pid, max_anonymous_bytes=0, min_available_bytes=0,
                 max_swap_used_bytes=0, on_limit=None, max_cgroup_full_pressure=0):
        self.pid = pid
        self.max_anonymous_bytes = max_anonymous_bytes
        self.min_available_bytes = min_available_bytes
        self.max_swap_used_bytes = max_swap_used_bytes
        self.on_limit = on_limit
        self.max_cgroup_full_pressure = max_cgroup_full_pressure
        self.breach_reason = None
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        start = time.monotonic()
        while not self.stop_event.is_set():
            try:
                pressure = cgroup_full_pressure(self.pid)
                if (pressure is not None and self.max_cgroup_full_pressure and
                        pressure > self.max_cgroup_full_pressure):
                    self.breach_reason = f'Cgroup memory full pressure exceeded {self.max_cgroup_full_pressure}%'
                    if self.on_limit:
                        self.on_limit()
                    break
                sample = counters(self.pid)
                sample['cgroup_memory_full_avg10'] = pressure
                sample['seconds'] = round(time.monotonic() - start, 3)
                self.samples.append(sample)
                if self.max_anonymous_bytes and sample['anonymous_bytes'] > self.max_anonymous_bytes:
                    self.breach_reason = 'Anonymous memory guard exceeded'
                elif self.min_available_bytes and sample['mem_available_bytes'] < self.min_available_bytes:
                    self.breach_reason = 'Minimum available RAM guard breached'
                elif self.max_swap_used_bytes and sample['system_swap_used_bytes'] > self.max_swap_used_bytes:
                    self.breach_reason = 'System swap-use guard exceeded'
                if self.breach_reason:
                    if self.on_limit:
                        self.on_limit()
                    break
            except (FileNotFoundError, ProcessLookupError):
                break
            self.stop_event.wait(0.2)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


class InferenceEngine(Protocol):
    process: subprocess.Popen
    lines: list

    def start(self): ...
    def wait_ready(self, timeout=180): ...
    def generate(self, max_tokens): ...
    def stop(self): ...


class LlamaCppAdapter:
    def __init__(self, binary, model, context, threads, port, load_mode,
                 batch_size=2048, ubatch_size=512, no_warmup=False, request_timeout=300,
                 fit='on', flash_attn='auto', numa=None, parallel=1):
        if not 1 <= parallel <= 4 or context < parallel * 512:
            raise ValueError('Parallel slots need at least 512 context tokens each')
        self.port = port
        self.request_timeout = request_timeout
        self.command = [str(binary), '--model', str(model), '--ctx-size', str(context),
                        '--threads', str(threads), '--threads-batch', str(threads),
                        '--parallel', str(parallel), '--host', '127.0.0.1', '--port', str(port),
                        '--no-webui', '--load-mode', load_mode,
                        '--batch-size', str(batch_size), '--ubatch-size', str(ubatch_size),
                        '--fit', fit, '--flash-attn', flash_attn]
        if no_warmup:
            self.command.append('--no-warmup')
        if numa:
            self.command.extend(['--numa', numa])
        self.process = None
        self.lines = []
        self.log_thread = None

    def start(self):
        started = time.monotonic()
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        bufsize=1)

        def read_log():
            for line in self.process.stdout:
                self.lines.append((time.monotonic(), line.rstrip()))

        self.log_thread = threading.Thread(target=read_log, daemon=True)
        self.log_thread.start()
        return started

    def wait_ready(self, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('Engine exited during startup: ' + '\n'.join(
                    line for _, line in self.lines[-15:]))
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/health', timeout=1) as response:
                    if response.status == 200:
                        return time.monotonic()
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.1)
        raise TimeoutError('Engine startup exceeded 180 seconds; inspect engine log')

    def generate(self, max_tokens):
        payload = json.dumps({'model': 'local', 'prompt': PROMPT,
                              'max_tokens': max_tokens, 'temperature': 0,
                              'stream': True,
                              'stream_options': {'include_usage': True}}).encode()
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.port}/v1/completions', data=payload,
            headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        first = None
        usage = None
        chunks = []
        finish_reason = None
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            for raw in response:
                if not raw.startswith(b'data: '):
                    continue
                content = raw[6:].strip()
                if content == b'[DONE]':
                    break
                event = json.loads(content)
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    if choice.get('finish_reason'):
                        finish_reason = choice['finish_reason']
                    token_text = choice.get('text', '')
                    if token_text:
                        if first is None:
                            first = time.monotonic()
                        chunks.append(token_text)
        end = time.monotonic()
        n_tokens = usage.get('completion_tokens') if usage else None
        decode_seconds = end - first if first is not None else None
        return {
            'time_to_first_token_seconds': first - start if first is not None else None,
            'request_seconds': end - start,
            'decode_seconds_after_first_token': decode_seconds,
            'completion_tokens': n_tokens,
            'tokens_per_second_after_first_token':
                (n_tokens - 1) / decode_seconds if n_tokens and n_tokens > 1 and decode_seconds > 0 else None,
            'response_chars': sum(map(len, chunks)),
            'response_sha256': hashlib.sha256(''.join(chunks).encode()).hexdigest(),
            'response_preview': ''.join(chunks)[:300],
            'finish_reason': finish_reason,
            'usage': usage,
        }

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_thread:
            self.log_thread.join(timeout=2)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def version(command):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return (completed.stdout or completed.stderr).splitlines()[0] if completed.returncode == 0 else None


def preflight(model, pressure_mib, oversized_mmap=False):
    if not model.is_file():
        raise FileNotFoundError(f'Model file missing: {model}')
    with model.open('rb') as model_file:
        magic = model_file.read(4)
    if model.suffix.lower() != '.gguf' or magic != b'GGUF':
        raise ValueError(f'Unsupported model format: {model}; expected GGUF')
    available = kib(fields('/proc/meminfo')['MemAvailable']) * 1024
    required = (2 * GIB if oversized_mmap else model.stat().st_size + 2 * GIB)
    required += pressure_mib * 1024 ** 2
    if available < required:
        raise MemoryError(f'Insufficient available RAM for safe baseline: {available / GIB:.2f} GiB '
                          f'available, {required / GIB:.2f} GiB required. '
                          'Do not infer safety from swap capacity.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-name', help='Name recorded in the benchmark artifact (default: model filename)')
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--model-quantization', required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--engine-revision', required=True)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--pressure-mib', type=int, default=0)
    parser.add_argument('--max-anonymous-mib', type=int, default=2048)
    parser.add_argument('--min-available-mib', type=int, default=1024)
    parser.add_argument('--max-swap-used-mib', type=int, default=1024)
    parser.add_argument('--max-cgroup-full-pressure', type=float, default=50.0,
                        help='Stop when the model cgroup spends more than this percent of time stalled on memory')
    parser.add_argument('--load-mode', choices=('auto', 'mmap', 'none'), default='auto')
    parser.add_argument('--oversized-mmap', action='store_true',
                        help='Explicitly permit a weight file larger than RAM under live memory guards')
    parser.add_argument('--no-warmup', action='store_true')
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--ubatch-size', type=int, default=512)
    parser.add_argument('--startup-timeout', type=int, default=180)
    parser.add_argument('--request-timeout', type=int, default=300)
    parser.add_argument('--fit', choices=('on', 'off'), default='on')
    parser.add_argument('--spec-type', choices=('none', 'ngram-simple'), default='none')
    parser.add_argument('--spec-draft-n-max', type=int, default=8)
    parser.add_argument('--flash-attn', choices=('on', 'off', 'auto'), default='auto')
    parser.add_argument('--numa', choices=('distribute', 'isolate', 'numactl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.context, args.max_tokens, args.threads, args.repetitions, args.max_anonymous_mib,
           args.min_available_mib, args.max_swap_used_mib, args.batch_size,
           args.ubatch_size, args.startup_timeout, args.request_timeout) < 1 or args.pressure_mib < 0 or not 0 < args.max_cgroup_full_pressure <= 100:
        parser.error('Limits must be positive; pressure allocation must be nonnegative; cgroup pressure limit must be at most 100%')
    if args.max_tokens >= args.context:
        parser.error('Generation limit must be smaller than context length')
    if args.ubatch_size > args.batch_size:
        parser.error('Microbatch size cannot exceed batch size')
    if args.oversized_mmap and args.load_mode != 'mmap':
        parser.error('Oversized-model test requires explicit --load-mode mmap')
    preflight(args.model, args.pressure_mib, args.oversized_mmap)
    if not args.server.is_file():
        raise FileNotFoundError(f'Engine binary missing: {args.server}')
    actual_sha256 = sha256(args.model)
    if actual_sha256 != args.model_sha256.lower():
        raise ValueError(f'Model SHA-256 mismatch: expected {args.model_sha256}, got {actual_sha256}')

    pressure = bytearray(args.pressure_mib * 1024 ** 2)
    for index in range(0, len(pressure), 4096):
        pressure[index] = 1
    engine: InferenceEngine = LlamaCppAdapter(args.server, args.model, args.context, args.threads,
                                              free_port(), args.load_mode, args.batch_size,
                                              args.ubatch_size, args.no_warmup, args.request_timeout,
                                              args.fit, args.flash_attn, args.numa)
    if args.spec_type != 'none':
        if not 1 <= args.spec_draft_n_max <= 64:
            parser.error('Speculative draft length must be between 1 and 64')
        engine.command.extend(['--spec-type', args.spec_type,
                               '--spec-draft-n-max', str(args.spec_draft_n_max)])
    result = {
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'engine': 'llama.cpp CPU',
        'engine_revision': args.engine_revision,
        'engine_binary': str(args.server.resolve()),
        'model': str(args.model.resolve()),
        'model_name': args.model_name or args.model.stem,
        'model_revision': args.model_revision,
        'model_quantization': args.model_quantization,
        'model_sha256': actual_sha256,
        'model_bytes': args.model.stat().st_size,
        'spec_type': args.spec_type,
        'spec_draft_n_max': args.spec_draft_n_max if args.spec_type != 'none' else 0,
        'context_tokens': args.context,
        'generation_limit_tokens': args.max_tokens,
        'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
        'threads': args.threads,
        'cpu_model': next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                           if line.startswith('model name')), None),
        'cpu_logical_count': os.cpu_count(),
        'os_release': next((line.split('=', 1)[1].strip().strip('"')
                            for line in Path('/etc/os-release').read_text().splitlines()
                            if line.startswith('PRETTY_NAME=')), None),
        'software_versions': {
            'python': sys.version.split()[0],
            'git': version(['git', '--version']),
            'cmake': version(['cmake', '--version']),
            'gcc': version(['gcc', '--version']),
            'engine': version([str(args.server), '--version']),
        },
        'pressure_mib': args.pressure_mib,
        'max_anonymous_mib': args.max_anonymous_mib,
        'min_available_mib': args.min_available_mib,
        'max_swap_used_mib': args.max_swap_used_mib,
        'max_cgroup_full_pressure': args.max_cgroup_full_pressure,
        'load_mode': args.load_mode,
        'oversized_mmap': args.oversized_mmap,
        'no_warmup': args.no_warmup,
        'batch_size': args.batch_size,
        'ubatch_size': args.ubatch_size,
        'startup_timeout_seconds': args.startup_timeout,
        'request_timeout_seconds': args.request_timeout,
        'fit': args.fit,
        'flash_attn': args.flash_attn,
        'numa': args.numa,
        'disk_free_bytes_before': shutil.disk_usage('/').free,
        'runs': [],
    }
    monitor = None
    try:
        start = engine.start()
        monitor = Monitor(engine.process.pid, args.max_anonymous_mib * 1024 ** 2,
                          args.min_available_mib * 1024 ** 2,
                          args.max_swap_used_mib * 1024 ** 2,
                          engine.process.terminate,
                          args.max_cgroup_full_pressure)
        monitor.start()
        ready = engine.wait_ready(args.startup_timeout)
        result['model_mapping_at_ready'] = model_mapping(engine.process.pid, args.model)
        result['startup_seconds'] = ready - start
        load_starts = [at for at, line in engine.lines if 'load_model: loading model' in line]
        load_ends = [at for at, line in engine.lines if 'llama_server: model loaded' in line]
        result['model_loading_seconds'] = (load_ends[0] - load_starts[0]
                                           if load_starts and load_ends else None)
        for _ in range(args.repetitions):
            result['runs'].append(engine.generate(args.max_tokens))
        result['model_mapping_after_runs'] = model_mapping(engine.process.pid, args.model)
        result['status'] = 'passed'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = ((monitor.breach_reason + '; engine stopped')
                           if monitor and monitor.breach_reason else f'{type(exc).__name__}: {exc}')
    finally:
        if monitor:
            monitor.stop()
            samples = monitor.samples
            result['samples'] = samples
            if samples:
                result['peak_rss_bytes'] = max(s['rss_high_water_bytes'] for s in samples)
                result['peak_virtual_bytes'] = max(s['virtual_bytes'] for s in samples)
                result['peak_anonymous_bytes'] = max(s['anonymous_bytes'] for s in samples)
                result['peak_file_pss_bytes'] = max(s['file_pss_bytes'] for s in samples)
                result['peak_process_swap_bytes'] = max(s['process_swap_bytes'] for s in samples)
                result['peak_system_swap_used_bytes'] = max(s['system_swap_used_bytes'] for s in samples)
                result['major_faults_delta'] = samples[-1]['major_faults'] - samples[0]['major_faults']
                result['disk_read_bytes_delta'] = samples[-1]['disk_read_bytes'] - samples[0]['disk_read_bytes']
        engine.stop()
        result['engine_log_start'] = [line for _, line in engine.lines[:20]]
        result['engine_log_tail'] = [line for _, line in engine.lines[-25:]]
        result['disk_free_bytes_after'] = shutil.disk_usage('/').free
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise RuntimeError(result['error'] + f'; see {args.output}')
    print(f"Benchmark saved: {args.output}")


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError, MemoryError) as exc:
        print(f'Benchmark preflight failed: {exc}', file=sys.stderr)
        raise SystemExit(2)
    except RuntimeError as exc:
        print(f'Benchmark run failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
