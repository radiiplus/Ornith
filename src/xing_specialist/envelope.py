"""Verify that small-model runs are inside the host resource scope."""

from pathlib import Path


GIB = 1024 ** 3


def check_values(quota, period, memory_high, memory_max, swap_max):
    if (period <= 0 or quota <= 0 or quota > period or
            memory_high <= 0 or memory_high > 4 * GIB or
            memory_max <= 0 or memory_max > 6 * GIB or
            swap_max < 0 or swap_max > 19 * GIB):
        raise RuntimeError('Small-model resource envelope is missing or too loose')


def require_small_model_envelope():
    try:
        group = Path('/proc/self/cgroup').read_text().split('::', 1)[1].strip().lstrip('/')
        scope = Path('/sys/fs/cgroup') / group
        quota, period = map(int, (scope / 'cpu.max').read_text().split())
        high = int((scope / 'memory.high').read_text())
        maximum = int((scope / 'memory.max').read_text())
        swap = int((scope / 'memory.swap.max').read_text())
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError('Run through scripts/run_dual_pipeline.sh or run_standalone_agent.sh') from exc
    check_values(quota, period, high, maximum, swap)
