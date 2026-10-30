"""Verifiable task stages and retry limits."""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class Step:
    key: str
    expected: str
    requires: tuple = ()


class Planner:
    def __init__(self, max_retries: int = 2):
        if max_retries < 0:
            raise ValueError('max_retries must be nonnegative')
        self.max_retries = max_retries
        self.failures = {}

    def plan(self, objective: str) -> tuple:
        if not objective.strip():
            raise ValueError('Objective is required')
        return (
            Step('inspect', 'Relevant files and test commands recorded'),
            Step('hypothesize', 'A falsifiable cause and expected change recorded', ('inspect',)),
            Step('change', 'A bounded patch applies cleanly', ('hypothesize',)),
            Step('verify', 'Approved tests pass with no regression', ('change',)),
            Step('record', 'Outcome linked to test evidence', ('verify',)),
        )

    @staticmethod
    def ready(step: Step, completed: set) -> bool:
        return set(step.requires).issubset(completed)

    def may_retry(self, action: str, evidence: str) -> bool:
        signature = sha256(action.encode()).hexdigest()
        evidence_signature = sha256(evidence.encode()).hexdigest()
        previous = self.failures.get(signature)
        if previous and previous[0] == evidence_signature:
            return False
        if previous and previous[1] >= self.max_retries:
            return False
        return True

    def failed(self, action: str, evidence: str) -> None:
        signature = sha256(action.encode()).hexdigest()
        evidence_signature = sha256(evidence.encode()).hexdigest()
        count = self.failures.get(signature, (None, 0))[1] + 1
        self.failures[signature] = (evidence_signature, count)
