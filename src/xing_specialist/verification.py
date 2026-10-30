"""Execute approved tests and record observed outcomes."""

from dataclasses import dataclass

from .memory import MemoryStore
from .tools import CommandResult, IsolatedExecutor


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    regressions: int
    evidence_ids: tuple
    results: tuple


class Verifier:
    def __init__(self, executor: IsolatedExecutor, store: MemoryStore):
        self.executor = executor
        self.store = store

    def verify(self, task_id: int, hypothesis_id: int, repository, commands: tuple,
               baseline: tuple = ()) -> VerificationReport:
        if not commands:
            raise ValueError('At least one verification command is required')
        if baseline and len(baseline) != len(commands):
            raise ValueError('Baseline and verification command counts differ')
        evidence = []
        results = []
        regressions = 0
        for index, command in enumerate(commands):
            result: CommandResult = self.executor.run(repository, command)
            regression = bool(baseline and baseline[index].exit_code == 0 and result.exit_code != 0)
            regressions += regression
            action_id = self.store.action(task_id, hypothesis_id, 'verification', ' '.join(command),
                                          result.stdout, result.stderr, result.exit_code,
                                          result.duration_seconds)
            evidence_id = self.store.test_result(task_id, action_id, ' '.join(command), 0,
                                                 result.exit_code, result.stdout, result.stderr,
                                                 regression=regression)
            self.store.observation(task_id, action_id,
                                   'Observed pass' if result.exit_code == 0 else 'Observed failure')
            evidence.append(evidence_id)
            results.append(result)
        return VerificationReport(all(result.exit_code == 0 for result in results),
                                  regressions, tuple(evidence), tuple(results))
