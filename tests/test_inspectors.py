from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import requests
from aidefense.request_handler import RequestHandler
from aidefense.exceptions import ApiError, ValidationError

from langchain_aidefense import Decision
from langchain_aidefense.inspectors import LLMInspector, MCPInspector, _make_config


def test_llm_inspector_retries_on_retryable_api_error():
    inspector = LLMInspector(
        api_key="x" * 64,
        endpoint="https://example.com",
        retry_total=2,
        retry_status_codes=[503],
        fail_open=False,
    )

    calls = {"count": 0}

    class Client:
        def inspect_conversation(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ApiError("boom", status_code=503)
            return SimpleNamespace(
                classifications=[],
                explanation=None,
                severity=None,
                rules=[],
                event_id=None,
                action="Allow",
                is_safe=True,
            )

    inspector._get_client = lambda: Client()  # type: ignore[method-assign]

    result = inspector.inspect_conversation([{"role": "user", "content": "hi"}], {})

    assert result == Decision.allow(raw_response=result.raw_response)
    assert calls["count"] == 2


def test_llm_inspector_does_not_retry_non_retryable_validation_error():
    inspector = LLMInspector(
        api_key="x" * 64,
        endpoint="https://example.com",
        retry_total=3,
        retry_status_codes=[503],
        fail_open=True,
    )

    calls = {"count": 0}

    class Client:
        def inspect_conversation(self, **kwargs):
            calls["count"] += 1
            raise ValidationError("bad request", status_code=400)

    inspector._get_client = lambda: Client()  # type: ignore[method-assign]

    result = inspector.inspect_conversation([{"role": "user", "content": "hi"}], {})

    assert result.action == "allow"
    assert calls["count"] == 1


def test_mcp_inspector_retries_on_http_503(monkeypatch: pytest.MonkeyPatch):
    inspector = MCPInspector(
        api_key="x" * 64,
        endpoint="https://example.com",
        retry_total=2,
        retry_status_codes=[503],
        fail_open=False,
    )

    calls = {"count": 0}

    class Response:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                error = requests.HTTPError("boom")
                error.response = self
                raise error

        def json(self):
            return self._payload

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return Response(503, {})
        return Response(200, {"result": {"is_safe": True, "action": "Allow", "classifications": []}})

    monkeypatch.setattr(inspector._session, "post", fake_post)

    result = inspector.inspect_request(tool_name="fetch_url", arguments={}, metadata={})

    assert result.action == "allow"
    assert calls["count"] == 2


def test_mcp_inspector_includes_metadata_in_payload(monkeypatch: pytest.MonkeyPatch):
    inspector = MCPInspector(
        api_key="x" * 64,
        endpoint="https://example.com",
        retry_total=1,
        fail_open=False,
    )

    seen_payloads: list[dict] = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"is_safe": True, "action": "Allow", "classifications": []}}

    def fake_post(*args, **kwargs):
        seen_payloads.append(kwargs["json"])
        return Response()

    monkeypatch.setattr(inspector._session, "post", fake_post)

    inspector.inspect_request(
        tool_name="fetch_url",
        arguments={"url": "https://example.com"},
        metadata={"src_app": "langchain", "dst_app": "fetch_url"},
    )
    inspector.inspect_response(
        tool_name="fetch_url",
        arguments={"url": "https://example.com"},
        result={"content": [{"type": "text", "text": "body"}]},
        metadata={"src_app": "langchain", "dst_app": "fetch_url"},
    )

    assert seen_payloads[0]["metadata"] == {"src_app": "langchain", "dst_app": "fetch_url"}
    assert seen_payloads[1]["metadata"] == {"src_app": "langchain", "dst_app": "fetch_url"}


def test_make_config_initializes_request_handler_compat_fields():
    config = _make_config(
        "https://example.com",
        5000,
        logging.getLogger("test"),
        pool_max_connections=5,
        pool_max_keepalive=7,
    )

    handler = RequestHandler(config)

    assert config.runtime_base_url == "https://example.com"
    assert config.timeout == 5
    assert hasattr(config, "connection_pool")
    assert handler._session is not None
