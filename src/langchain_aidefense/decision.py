from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Decision:
    action: str
    reasons: list[str] = field(default_factory=list)
    raw_response: Any = None
    severity: str | None = None
    classifications: list[str] | None = None
    rules: list[Any] | None = None
    explanation: str | None = None
    event_id: str | None = None

    def allows(self) -> bool:
        return self.action != "block"

    @classmethod
    def allow(
        cls,
        reasons: list[str] | None = None,
        *,
        raw_response: Any = None,
        severity: str | None = None,
        classifications: list[str] | None = None,
        rules: list[Any] | None = None,
        explanation: str | None = None,
        event_id: str | None = None,
    ) -> "Decision":
        return cls(
            action="allow",
            reasons=reasons or [],
            raw_response=raw_response,
            severity=severity,
            classifications=classifications,
            rules=rules,
            explanation=explanation,
            event_id=event_id,
        )

    @classmethod
    def block(
        cls,
        reasons: list[str],
        *,
        raw_response: Any = None,
        severity: str | None = None,
        classifications: list[str] | None = None,
        rules: list[Any] | None = None,
        explanation: str | None = None,
        event_id: str | None = None,
    ) -> "Decision":
        return cls(
            action="block",
            reasons=reasons,
            raw_response=raw_response,
            severity=severity,
            classifications=classifications,
            rules=rules,
            explanation=explanation,
            event_id=event_id,
        )

