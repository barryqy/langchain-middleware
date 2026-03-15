from __future__ import annotations

from .decision import Decision


class SecurityPolicyError(Exception):
    def __init__(self, decision: Decision, message: str | None = None) -> None:
        self.decision = decision
        self.message = message or self._format_message(decision)
        super().__init__(self.message)

    @staticmethod
    def _format_message(decision: Decision) -> str:
        if decision.explanation:
            return decision.explanation
        if decision.reasons:
            return "; ".join(decision.reasons)
        return "Security policy violation"

    def __str__(self) -> str:
        return self.message

