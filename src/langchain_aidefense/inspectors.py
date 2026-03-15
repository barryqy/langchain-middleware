from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import asdict, is_dataclass
from typing import Any

import requests
from aidefense import ChatInspectionClient, Config
from aidefense.exceptions import ApiError
from aidefense.runtime.chat_models import Message, Role
from aidefense.runtime.models import InspectionConfig, Metadata, Rule, RuleName
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .decision import Decision


logger = logging.getLogger("langchain_aidefense.inspectors")


class _StandaloneConfig(Config):
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


def _make_config(
    runtime_base_url: str,
    timeout_ms: int | None,
    custom_logger: logging.Logger,
    *,
    pool_max_connections: int | None,
    pool_max_keepalive: int | None,
) -> Config:
    config = _StandaloneConfig()
    config._initialize(
        runtime_base_url=runtime_base_url,
        timeout=None if timeout_ms is None else max(1, int(timeout_ms / 1000)),
        logger=custom_logger,
        retry_config={
            # The middleware owns retry policy, so keep the SDK transport single-shot.
            "total": 0,
            "backoff_factor": 0,
            "status_forcelist": [],
        },
        pool_config=_pool_config(
            pool_max_connections=pool_max_connections,
            pool_max_keepalive=pool_max_keepalive,
        ),
    )
    config.runtime_base_url = runtime_base_url.rstrip("/")
    return config


class LLMInspector:
    def __init__(
        self,
        *,
        api_key: str | None,
        endpoint: str | None,
        default_rules: list[str] | None = None,
        entity_types: list[str] | None = None,
        timeout_ms: int | None = None,
        retry_total: int | None = None,
        retry_backoff: float | None = None,
        retry_status_codes: list[int] | None = None,
        pool_max_connections: int | None = None,
        pool_max_keepalive: int | None = None,
        fail_open: bool = True,
        logger_instance: logging.Logger | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.endpoint = (endpoint or "").rstrip("/") or None
        self.default_rules = list(default_rules or [])
        self.entity_types = list(entity_types or [])
        self.timeout_ms = timeout_ms
        self.retry_total = max(1, retry_total or 1)
        self.retry_backoff = max(0.0, retry_backoff or 0.0)
        self.retry_status_codes = tuple(retry_status_codes or (429, 500, 502, 503, 504))
        self.pool_max_connections = pool_max_connections
        self.pool_max_keepalive = pool_max_keepalive
        self.fail_open = fail_open
        self.logger = logger_instance or logger
        self._client: ChatInspectionClient | None = None

    def inspect_conversation(self, messages: list[dict[str, Any]], metadata: dict[str, Any]) -> Decision:
        if not self.endpoint or not self.api_key:
            return Decision.allow()

        def call() -> Any:
            client = self._get_client()
            return client.inspect_conversation(
                messages=_messages_to_runtime(messages),
                metadata=_metadata_to_runtime(metadata),
                config=_inspection_config(self.default_rules, self.entity_types),
                timeout=_timeout_seconds(self.timeout_ms),
            )

        return self._run_with_retry(call, kind="llm")

    async def ainspect_conversation(self, messages: list[dict[str, Any]], metadata: dict[str, Any]) -> Decision:
        return await asyncio.to_thread(self.inspect_conversation, messages, metadata)

    def close(self) -> None:
        self._client = None

    async def aclose(self) -> None:
        self.close()

    def _get_client(self) -> ChatInspectionClient:
        if self._client is None:
            config = _make_config(
                self.endpoint or "",
                self.timeout_ms,
                self.logger,
                pool_max_connections=self.pool_max_connections,
                pool_max_keepalive=self.pool_max_keepalive,
            )
            self._client = ChatInspectionClient(api_key=self.api_key, config=config)
        return self._client

    def _run_with_retry(self, call: Any, *, kind: str) -> Decision:
        last_error: Exception | None = None
        for attempt in range(self.retry_total):
            try:
                response = call()
                return _decision_from_inspect_response(response)
            except Exception as exc:  # defensive on SDK/runtime mismatches
                last_error = exc
                if attempt >= self.retry_total - 1 or not self._should_retry(exc):
                    break
                if self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (2 ** attempt))

        if self.fail_open:
            reason = f"{kind} inspection error: {last_error}"
            self.logger.warning("%s", reason)
            return Decision.allow([reason], raw_response=last_error)

        raise last_error  # type: ignore[misc]

    def _should_retry(self, error: Exception) -> bool:
        if isinstance(error, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(error, requests.HTTPError):
            response = getattr(error, "response", None)
            return bool(response is not None and response.status_code in self.retry_status_codes)
        if isinstance(error, ApiError):
            return bool(error.status_code in self.retry_status_codes)
        return False


class MCPInspector:
    def __init__(
        self,
        *,
        api_key: str | None,
        endpoint: str | None,
        timeout_ms: int | None = None,
        retry_total: int | None = None,
        retry_backoff: float | None = None,
        retry_status_codes: list[int] | None = None,
        pool_max_connections: int | None = None,
        pool_max_keepalive: int | None = None,
        fail_open: bool = True,
        logger_instance: logging.Logger | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key
        self.endpoint = (endpoint or "").rstrip("/") or None
        self.timeout_ms = timeout_ms
        self.retry_total = max(1, retry_total or 1)
        self.retry_backoff = max(0.0, retry_backoff or 0.0)
        self.retry_status_codes = tuple(retry_status_codes or (429, 500, 502, 503, 504))
        self.pool_max_connections = pool_max_connections
        self.pool_max_keepalive = pool_max_keepalive
        self.fail_open = fail_open
        self.logger = logger_instance or logger
        self._counter = itertools.count(1)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
            }
        )
        adapter = HTTPAdapter(
            pool_connections=(self.pool_max_connections or 10),
            pool_maxsize=(self.pool_max_keepalive or self.pool_max_connections or 20),
            max_retries=Retry(total=0, raise_on_status=False),
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def inspect_request(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": _request_params(method, tool_name, arguments),
            "id": next(self._counter),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return self._post(payload, context="tool_request")

    def inspect_response(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        metadata: dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": _request_params(method, tool_name, arguments),
            "result": _result_to_content_dict(result),
            "id": next(self._counter),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return self._post(payload, context="tool_response")

    async def ainspect_request(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        metadata: dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        return await asyncio.to_thread(
            self.inspect_request,
            tool_name=tool_name,
            arguments=arguments,
            metadata=metadata,
            method=method,
        )

    async def ainspect_response(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        metadata: dict[str, Any],
        method: str = "tools/call",
    ) -> Decision:
        return await asyncio.to_thread(
            self.inspect_response,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            metadata=metadata,
            method=method,
        )

    def close(self) -> None:
        self._session.close()

    async def aclose(self) -> None:
        self.close()

    def _post(self, payload: dict[str, Any], *, context: str) -> Decision:
        if not self.endpoint or not self.api_key:
            return Decision.allow()

        url = f"{self.endpoint}/api/v1/inspect/mcp"
        headers = {"X-Cisco-AI-Defense-API-Key": self.api_key}
        last_error: Exception | None = None

        for attempt in range(self.retry_total):
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=_timeout_seconds(self.timeout_ms),
                )
                response.raise_for_status()
                return _decision_from_mcp_payload(response.json())
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_total - 1 or not self._should_retry(exc):
                    break
                if self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (2 ** attempt))

        if self.fail_open:
            reason = f"{context} inspection error: {last_error}"
            self.logger.warning("%s", reason)
            return Decision.allow([reason], raw_response=last_error)

        raise last_error  # type: ignore[misc]

    def _should_retry(self, error: Exception) -> bool:
        if isinstance(error, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(error, requests.HTTPError):
            response = getattr(error, "response", None)
            return bool(response is not None and response.status_code in self.retry_status_codes)
        return False


def _messages_to_runtime(messages: list[dict[str, Any]]) -> list[Message]:
    out: list[Message] = []
    for item in messages:
        role = str(item.get("role", "user")).lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        out.append(Message(role=Role(role), content=str(item.get("content", ""))))
    return out


def _metadata_to_runtime(metadata: dict[str, Any]) -> Metadata | None:
    if not metadata:
        return None

    known_keys = {
        "user",
        "created_at",
        "src_app",
        "dst_app",
        "sni",
        "dst_ip",
        "src_ip",
        "dst_host",
        "user_agent",
        "client_transaction_id",
    }
    kwargs = {key: value for key, value in metadata.items() if key in known_keys and value is not None}
    if not kwargs:
        return None
    return Metadata(**kwargs)


def _inspection_config(default_rules: list[str], entity_types: list[str]) -> InspectionConfig | None:
    if not default_rules and not entity_types:
        return None

    rules: list[Rule] = []
    for item in default_rules:
        try:
            rule_name = RuleName(item)
        except ValueError:
            continue
        rules.append(Rule(rule_name=rule_name, entity_types=entity_types or None))

    if not rules and entity_types:
        rules.append(Rule(rule_name=RuleName.PII, entity_types=entity_types))

    if not rules:
        return None
    return InspectionConfig(enabled_rules=rules)


def _pool_config(
    *,
    pool_max_connections: int | None,
    pool_max_keepalive: int | None,
) -> dict[str, int]:
    # requests' adapter doesn't expose keepalive as a first-class setting, so we map the
    # Cisco-style knobs to the closest pool sizes it does support.
    return {
        "pool_connections": pool_max_connections or 10,
        "pool_maxsize": pool_max_keepalive or pool_max_connections or 20,
    }


def _decision_from_inspect_response(response: Any) -> Decision:
    classifications = [_enum_value(item) for item in getattr(response, "classifications", []) or []]
    reasons = list(classifications)

    explanation = getattr(response, "explanation", None)
    if explanation and explanation not in reasons:
        reasons.append(explanation)

    kwargs = {
        "raw_response": response,
        "severity": _enum_value(getattr(response, "severity", None)),
        "classifications": classifications or None,
        "rules": [_rule_to_dict(item) for item in getattr(response, "rules", []) or []] or None,
        "explanation": explanation,
        "event_id": getattr(response, "event_id", None),
    }

    action_name = _enum_name(getattr(response, "action", None))
    is_safe = bool(getattr(response, "is_safe", True))
    if action_name == "BLOCK" or not is_safe:
        return Decision.block(reasons or ["policy violation"], **kwargs)

    return Decision.allow(reasons or [], **kwargs)


def _decision_from_mcp_payload(payload: dict[str, Any]) -> Decision:
    if payload.get("error"):
        error = payload["error"]
        reason = error.get("message", "MCP inspection error")
        return Decision.block([reason], raw_response=payload, explanation=reason)

    result = payload.get("result", payload)
    if isinstance(result, dict) and "is_safe" in result:
        return _decision_from_result_dict(result, raw_response=payload)

    return Decision.allow(raw_response=payload)


def _decision_from_result_dict(result: dict[str, Any], *, raw_response: Any) -> Decision:
    classifications = [_enum_value(item) for item in result.get("classifications", [])]
    reasons = list(classifications)
    explanation = result.get("explanation")
    if explanation and explanation not in reasons:
        reasons.append(explanation)

    action_name = _enum_name(result.get("action"))
    is_safe = bool(result.get("is_safe", True))
    kwargs = {
        "raw_response": raw_response,
        "severity": _enum_value(result.get("severity")),
        "classifications": classifications or None,
        "rules": [_rule_to_dict(item) for item in result.get("rules", [])] or None,
        "explanation": explanation,
        "event_id": result.get("event_id"),
    }

    if action_name == "BLOCK" or not is_safe:
        return Decision.block(reasons or ["policy violation"], **kwargs)
    return Decision.allow(reasons or [], **kwargs)


def _request_params(method: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if method == "resources/read":
        return {"uri": tool_name}
    return {"name": tool_name, "arguments": arguments or {}}


def _result_to_content_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and "content" in result:
        return result
    if isinstance(result, list):
        return {"content": result}
    if isinstance(result, dict):
        return {"content": [result]}
    if isinstance(result, str):
        return {"content": [{"type": "text", "text": result}]}
    if is_dataclass(result):
        return {"content": [asdict(result)]}
    return {"content": [{"type": "text", "text": str(result)}]}


def _timeout_seconds(timeout_ms: int | None) -> int | None:
    if timeout_ms is None:
        return None
    return max(1, int(timeout_ms / 1000))


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(value, str):
        return value.upper()
    return str(value).upper()


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    if raw is None:
        return None
    return str(raw)


def _rule_to_dict(rule: Any) -> Any:
    if is_dataclass(rule):
        return asdict(rule)
    return rule
