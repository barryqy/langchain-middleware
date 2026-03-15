from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping


VALID_MODES = {"off", "monitor", "enforce"}
VALID_VIOLATION_BEHAVIORS = {"error", "end", "replace"}


def _normalize_mode(value: str | None, *, default: str) -> str:
    if value is None:
        return default

    mode = value.strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported mode: {value!r}. Expected one of {sorted(VALID_MODES)}.")

    return mode


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Unsupported boolean value: {value!r}.")


def _parse_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _parse_list(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()

    tmp = value.strip()
    if tmp.startswith("["):
        parsed = json.loads(tmp)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON list, got {type(parsed).__name__}.")
        return tuple(str(item) for item in parsed if str(item).strip())

    return tuple(item.strip() for item in tmp.split(",") if item.strip())


def _parse_int_list(value: str | None) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_list(value))


def _normalize_runtime_endpoint(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None

    endpoint = value.strip().rstrip("/")
    suffixes = (
        "/api/v1/inspect/chat",
        "/api/v1/inspect/mcp",
        "/api",
    )

    for suffix in suffixes:
        if endpoint.endswith(suffix):
            return endpoint[: -len(suffix)]

    return endpoint


def _normalize_violation_behavior(value: str | None, *, default: str = "error") -> str:
    if value is None:
        return default

    behavior = value.strip().lower()
    if behavior not in VALID_VIOLATION_BEHAVIORS:
        expected = sorted(VALID_VIOLATION_BEHAVIORS)
        raise ValueError(f"Unsupported violation behavior: {value!r}. Expected one of {expected}.")

    return behavior


@dataclass(frozen=True)
class EndpointSettings:
    mode: str
    endpoint: str | None
    api_key: str | None
    fail_open: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def validate(self, label: str) -> None:
        if not self.enabled:
            return

        if not self.endpoint:
            raise ValueError(f"{label} inspection is enabled but no endpoint is configured.")

        if not self.api_key:
            raise ValueError(f"{label} inspection is enabled but no API key is configured.")


@dataclass(frozen=True)
class AIDefenseSettings:
    llm: EndpointSettings
    tools: EndpointSettings
    timeout_ms: int | None = None
    retry_total: int | None = None
    retry_backoff: float | None = None
    retry_status_codes: tuple[int, ...] = ()
    pool_max_connections: int | None = None
    pool_max_keepalive: int | None = None
    llm_default_rules: tuple[str, ...] = ()
    llm_entity_types: tuple[str, ...] = ()
    violation_behavior: str = "error"
    violation_message: str | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        validate: bool = True,
    ) -> "AIDefenseSettings":
        values = dict(os.environ if env is None else env)

        llm_mode = _normalize_mode(values.get("AGENTSEC_API_MODE_LLM"), default="monitor")
        tool_mode = _normalize_mode(values.get("AGENTSEC_API_MODE_MCP"), default=llm_mode)

        llm_endpoint = _normalize_runtime_endpoint(values.get("AI_DEFENSE_API_MODE_LLM_ENDPOINT"))
        llm_api_key = values.get("AI_DEFENSE_API_MODE_LLM_API_KEY")

        tool_endpoint = _normalize_runtime_endpoint(
            values.get("AI_DEFENSE_API_MODE_MCP_ENDPOINT") or values.get("AI_DEFENSE_API_MODE_LLM_ENDPOINT")
        )
        tool_api_key = values.get("AI_DEFENSE_API_MODE_MCP_API_KEY") or values.get("AI_DEFENSE_API_MODE_LLM_API_KEY")

        settings = cls(
            llm=EndpointSettings(
                mode=llm_mode,
                endpoint=llm_endpoint,
                api_key=llm_api_key,
                fail_open=_parse_bool(values.get("AGENTSEC_API_MODE_FAIL_OPEN_LLM"), default=True),
            ),
            tools=EndpointSettings(
                mode=tool_mode,
                endpoint=tool_endpoint,
                api_key=tool_api_key,
                fail_open=_parse_bool(values.get("AGENTSEC_API_MODE_FAIL_OPEN_MCP"), default=True),
            ),
            timeout_ms=_ms_from_seconds(_parse_int(values.get("AGENTSEC_TIMEOUT"))),
            retry_total=_parse_int(values.get("AGENTSEC_RETRY_TOTAL")),
            retry_backoff=_parse_float(values.get("AGENTSEC_RETRY_BACKOFF_FACTOR")),
            retry_status_codes=_parse_int_list(values.get("AGENTSEC_RETRY_STATUS_FORCELIST")),
            pool_max_connections=_parse_int(values.get("AGENTSEC_POOL_MAX_CONNECTIONS")),
            pool_max_keepalive=_parse_int(values.get("AGENTSEC_POOL_MAX_KEEPALIVE")),
            llm_default_rules=_parse_list(values.get("AGENTSEC_LLM_RULES")),
            llm_entity_types=_parse_list(values.get("AGENTSEC_LLM_ENTITY_TYPES")),
            violation_behavior=_normalize_violation_behavior(values.get("AGENTSEC_VIOLATION_BEHAVIOR")),
            violation_message=values.get("AGENTSEC_VIOLATION_MESSAGE"),
        )

        if validate:
            settings.validate()
        return settings

    def validate(self) -> None:
        self.llm.validate("LLM")
        self.tools.validate("tool")
        _normalize_violation_behavior(self.violation_behavior)


def _ms_from_seconds(value: int | None) -> int | None:
    if value is None:
        return None
    return value * 1000
