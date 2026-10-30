"""Fail-closed command policy and isolated execution for untrusted repositories."""

from dataclasses import dataclass
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import tempfile
import time

from .memory import redact


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    command: tuple
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timed_out: bool = False


class CommandPolicy:
    @staticmethod
    def approve(command: tuple) -> tuple:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ToolError('Command must be a nonempty argument list')
        name = command[0]
        args = command[1:]
        if name in ('python3', 'python') and len(args) >= 2 and args[:2] == ('-m', 'unittest'):
            return command
        if name == 'pytest' and all(not arg.startswith('--override-ini') for arg in args):
            return command
        if name == 'make' and args == ('test',):
            return command
        if name == 'cargo' and args == ('test', '--offline'):
            return command
        if name == 'git' and args in (('diff', '--check'), ('status', '--short')):
            return command
        if name == 'ruff' and args and args[0] == 'check':
            return command
        raise ToolError('Command is not approved by the execution policy: ' + shlex.join(command))


class IsolatedExecutor:
    def __init__(self, state_dir: Path, timeout_seconds: int = 60,
                 memory_mib: int = 2048, max_repo_mib: int = 50):
        if min(timeout_seconds, memory_mib, max_repo_mib) < 1:
            raise ValueError('Resource limits must be positive')
        if not shutil.which('bwrap'):
            raise ToolError('Bubblewrap is required for untrusted command execution')
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.memory_mib = memory_mib
        self.max_repo_bytes = max_repo_mib * 1024 ** 2

    def _copy_repository(self, source: Path, destination: Path):
        source = source.resolve(strict=True)
        if not source.is_dir():
            raise ToolError('Repository path must be a directory')
        total = 0
        count = 0
        excluded = {'.git', '.aws', '.env', '.venv', 'node_modules', 'vendor', 'models',
                    'artifacts', '__pycache__', '.pytest_cache'}
        for base, dirs, files in os.walk(source):
            dirs[:] = [name for name in dirs if name not in excluded]
            relative = Path(base).relative_to(source)
            target = destination / relative
            target.mkdir(parents=True, exist_ok=True)
            for name in files:
                path = Path(base) / name
                if path.is_symlink():
                    raise ToolError('Symlinks are not accepted in isolated repository copies')
                if name.endswith(('.pem', '.key', '.sqlite3', '.gguf')) or name.startswith('.env'):
                    continue
                if not path.is_file():
                    raise ToolError('Unsupported repository file type')
                count += 1
                total += path.stat().st_size
                if count > 3000 or total > self.max_repo_bytes:
                    raise ToolError('Repository exceeds isolated copy budget')
                shutil.copy2(path, target / name)

    def run(self, repository: Path, command: tuple) -> CommandResult:
        command = CommandPolicy.approve(tuple(command))
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='xing-sandbox-', dir=self.state_dir) as scratch:
            workspace = Path(scratch) / 'workspace'
            self._copy_repository(Path(repository), workspace)
            sandbox_command = [
                'bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
                '--ro-bind', '/usr', '/usr', '--ro-bind', '/bin', '/bin',
                '--ro-bind', '/lib64', '/lib64', '--proc', '/proc', '--dev', '/dev',
                '--tmpfs', '/tmp', '--bind', str(workspace), '/workspace',
                '--chdir', '/workspace', '--setenv', 'HOME', '/tmp',
                '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'PYTHONPATH', '/workspace/src',
                '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', '--setenv', 'LANG', 'C.UTF-8',
                '--', *command,
            ]

            def limits():
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_mib * 1024 ** 2,) * 2)
                resource.setrlimit(resource.RLIMIT_CPU, (self.timeout_seconds + 1,) * 2)
                resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 ** 2,) * 2)

            try:
                completed = subprocess.run(sandbox_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, timeout=self.timeout_seconds, preexec_fn=limits)
                return CommandResult(command, redact(completed.stdout), redact(completed.stderr),
                                     completed.returncode, time.monotonic() - started)
            except subprocess.TimeoutExpired as exc:
                return CommandResult(command, redact((exc.stdout or b'').decode(errors='replace')),
                                     redact((exc.stderr or b'').decode(errors='replace')) + '\nTimed out',
                                     124, time.monotonic() - started, True)
