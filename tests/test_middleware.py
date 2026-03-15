from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain_aidefense import AIDefenseMiddleware, Decision, SecurityPolicyError
from langchain_aidefense.config import AIDefenseSettings, EndpointSettings


def allow_decision() -> Decision:
    return Decision.allow()


def block_decision(reason: str = "blocked") -> Decision:
    return Decision.block([reason], explanation=reason)


@dataclass
class StubLLMInspector:
    request_decision: Decision = field(default_factory=allow_decision)
    response_decision: Decision = field(default_factory=allow_decision)

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]], dict[str, str]]] = []

    def inspect_conversation(self, messages, metadata):
        phase = "response" if messages and messages[0]["role"] == "assistant" else "request"
        self.calls.append((phase, messages, metadata))
        return self.response_decision if phase == "response" else self.request_decision

    async def ainspect_conversation(self, messages, metadata):
        return self.inspect_conversation(messages, metadata)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@dataclass
class StubMCPInspector:
    request_decision: Decision = field(default_factory=allow_decision)
    response_decision: Decision = field(default_factory=allow_decision)

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, str, dict, str]] = []

    def inspect_request(self, tool_name, arguments, metadata, method="tools/call"):
        self.calls.append(("request", tool_name, arguments, method))
        return self.request_decision

    def inspect_response(self, tool_name, arguments, result, metadata, method="tools/call"):
        self.calls.append(("response", tool_name, arguments, method))
        return self.response_decision

    async def ainspect_request(self, tool_name, arguments, metadata, method="tools/call"):
        return self.inspect_request(tool_name, arguments, metadata, method)

    async def ainspect_response(self, tool_name, arguments, result, metadata, method="tools/call"):
        return self.inspect_response(tool_name, arguments, result, metadata, method)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def make_settings(
    *,
    llm_mode: str = "monitor",
    tool_mode: str = "monitor",
    violation_behavior: str = "error",
    violation_message: str | None = None,
) -> AIDefenseSettings:
    return AIDefenseSettings(
        llm=EndpointSettings(
            mode=llm_mode,
            endpoint="https://example.com",
            api_key="llm-key",
            fail_open=True,
        ),
        tools=EndpointSettings(
            mode=tool_mode,
            endpoint="https://example.com",
            api_key="tool-key",
            fail_open=True,
        ),
        violation_behavior=violation_behavior,
        violation_message=violation_message,
    )


def make_model_request() -> ModelRequest:
    messages = [
        HumanMessage(content="Hello there"),
    ]
    return ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        system_message=SystemMessage(content="Be helpful."),
        messages=messages,
        state={"messages": messages},
        runtime=SimpleNamespace(),
    )


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL."""
    return f"fetched {url}"


def make_tool_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "id": "call-1",
            "name": "fetch_url",
            "args": {"url": "https://example.com"},
        },
        tool=fetch_url,
        state={"messages": []},
        runtime=SimpleNamespace(),
    )


def test_wrap_model_call_inspects_request_and_response():
    llm = StubLLMInspector()
    middleware = AIDefenseMiddleware(
        settings=make_settings(tool_mode="off"),
        llm_inspector=llm,
        mcp_inspector=StubMCPInspector(),
    )

    response = middleware.wrap_model_call(
        make_model_request(),
        lambda request: ModelResponse(result=[AIMessage(content="All clear")]),
    )

    assert isinstance(response, ModelResponse)
    assert [phase for phase, *_ in llm.calls] == ["request", "response"]
    assert llm.calls[0][1][0] == {"role": "system", "content": "Be helpful."}
    assert llm.calls[0][1][1] == {"role": "user", "content": "Hello there"}


def test_wrap_model_call_blocks_in_enforce_mode_before_handler():
    llm = StubLLMInspector(request_decision=block_decision("bad prompt"))
    middleware = AIDefenseMiddleware(
        settings=make_settings(llm_mode="enforce", tool_mode="off"),
        llm_inspector=llm,
        mcp_inspector=StubMCPInspector(),
    )

    called = False

    def handler(_: ModelRequest) -> ModelResponse:
        nonlocal called
        called = True
        return ModelResponse(result=[AIMessage(content="should not happen")])

    with pytest.raises(SecurityPolicyError, match="llm_request"):
        middleware.wrap_model_call(make_model_request(), handler)

    assert called is False


def test_wrap_model_call_ends_with_violation_message():
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="enforce",
            tool_mode="off",
            violation_behavior="end",
        ),
        llm_inspector=StubLLMInspector(request_decision=block_decision("bad prompt")),
        mcp_inspector=StubMCPInspector(),
    )

    called = False

    def handler(_: ModelRequest) -> ModelResponse:
        nonlocal called
        called = True
        return ModelResponse(result=[AIMessage(content="should not happen")])

    result = middleware.wrap_model_call(make_model_request(), handler)

    assert isinstance(result, AIMessage)
    assert result.content == "AI Defense blocked llm_request: bad prompt"
    assert called is False


def test_wrap_model_call_replaces_last_request_message():
    seen_messages: list[BaseMessage] = []
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="enforce",
            tool_mode="off",
            violation_behavior="replace",
        ),
        llm_inspector=StubLLMInspector(request_decision=block_decision("bad prompt")),
        mcp_inspector=StubMCPInspector(),
    )

    def handler(request: ModelRequest) -> ModelResponse:
        seen_messages.extend(request.messages)
        return ModelResponse(result=[AIMessage(content="safe")])

    result = middleware.wrap_model_call(make_model_request(), handler)

    assert isinstance(result, ModelResponse)
    assert seen_messages[-1].content == "AI Defense blocked llm_request: bad prompt"


def test_wrap_model_call_replace_rewrites_system_and_request_messages():
    seen: dict[str, Any] = {}
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="enforce",
            tool_mode="off",
            violation_behavior="replace",
        ),
        llm_inspector=StubLLMInspector(request_decision=block_decision("bad prompt")),
        mcp_inspector=StubMCPInspector(),
    )
    request = ModelRequest(
        model=FakeListChatModel(responses=["unused"]),
        system_message=SystemMessage(content="hidden system prompt"),
        messages=[HumanMessage(content="safe user prompt")],
        state={"messages": [HumanMessage(content="safe user prompt")]},
        runtime=SimpleNamespace(),
    )

    def handler(current_request: ModelRequest) -> ModelResponse:
        seen["system"] = current_request.system_message.content if current_request.system_message else None
        seen["messages"] = [message.content for message in current_request.messages]
        seen["state_messages"] = [message.content for message in current_request.state["messages"]]
        return ModelResponse(result=[AIMessage(content="safe")])

    middleware.wrap_model_call(request, handler)

    assert seen["system"] == "AI Defense blocked llm_request: bad prompt"
    assert seen["messages"] == ["AI Defense blocked llm_request: bad prompt"]
    assert seen["state_messages"] == ["AI Defense blocked llm_request: bad prompt"]


@pytest.mark.asyncio
async def test_awrap_model_call_uses_async_inspector():
    llm = StubLLMInspector()
    middleware = AIDefenseMiddleware(
        settings=make_settings(tool_mode="off"),
        llm_inspector=llm,
        mcp_inspector=StubMCPInspector(),
    )

    result = await middleware.awrap_model_call(
        make_model_request(),
        lambda request: _return_model_response(request),
    )

    assert isinstance(result, ModelResponse)
    assert [phase for phase, *_ in llm.calls] == ["request", "response"]


async def _return_model_response(_: ModelRequest) -> ModelResponse:
    return ModelResponse(result=[AIMessage(content="Async ok")])


def test_wrap_model_call_replace_preserves_extended_model_response_shape():
    command = Command(update={"flag": True})
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="enforce",
            tool_mode="off",
            violation_behavior="replace",
        ),
        llm_inspector=StubLLMInspector(response_decision=block_decision("bad output")),
        mcp_inspector=StubMCPInspector(),
    )

    result = middleware.wrap_model_call(
        make_model_request(),
        lambda request: ExtendedModelResponse(
            model_response=ModelResponse(
                result=[AIMessage(content="unsafe output")],
                structured_response={"answer": "unsafe"},
            ),
            command=command,
        ),
    )

    assert isinstance(result, ExtendedModelResponse)
    assert result.command is command
    assert result.model_response.structured_response == {"answer": "unsafe"}
    assert result.model_response.result[0].content == "AI Defense blocked llm_response: bad output"


def test_wrap_tool_call_blocks_before_execution():
    mcp = StubMCPInspector(request_decision=block_decision("tool blocked"))
    middleware = AIDefenseMiddleware(
        settings=make_settings(llm_mode="off", tool_mode="enforce"),
        llm_inspector=StubLLMInspector(),
        mcp_inspector=mcp,
    )

    called = False

    def handler(_: ToolCallRequest):
        nonlocal called
        called = True
        return ToolMessage(content="nope", tool_call_id="call-1")

    with pytest.raises(SecurityPolicyError, match="tool_request"):
        middleware.wrap_tool_call(make_tool_request(), handler)

    assert called is False


def test_wrap_tool_call_replaces_blocked_request_with_tool_message():
    mcp = StubMCPInspector(request_decision=block_decision("tool blocked"))
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="off",
            tool_mode="enforce",
            violation_behavior="replace",
        ),
        llm_inspector=StubLLMInspector(),
        mcp_inspector=mcp,
    )

    called = False

    def handler(_: ToolCallRequest):
        nonlocal called
        called = True
        return ToolMessage(content="nope", tool_call_id="call-1")

    result = middleware.wrap_tool_call(make_tool_request(), handler)

    assert isinstance(result, ToolMessage)
    assert result.content == "AI Defense blocked tool_request: tool blocked"
    assert result.status == "error"
    assert called is False


def test_wrap_tool_call_end_marks_state_for_exit():
    mcp = StubMCPInspector(request_decision=block_decision("tool blocked"))
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="off",
            tool_mode="enforce",
            violation_behavior="end",
        ),
        llm_inspector=StubLLMInspector(),
        mcp_inspector=mcp,
    )

    result = middleware.wrap_tool_call(make_tool_request(), lambda request: ToolMessage(content="ok", tool_call_id=request.tool_call["id"]))

    assert isinstance(result, Command)
    assert result.update["aidefense_pending_end_message"] == "AI Defense blocked tool_request: tool blocked"


def test_constructor_skips_tool_validation_when_tool_hooks_disabled():
    settings = AIDefenseSettings(
        llm=EndpointSettings(mode="off", endpoint=None, api_key=None, fail_open=True),
        tools=EndpointSettings(mode="monitor", endpoint=None, api_key=None, fail_open=True),
    )

    middleware = AIDefenseMiddleware(
        settings=settings,
        inspect_tool_requests=False,
        inspect_tool_responses=False,
        llm_inspector=StubLLMInspector(),
        mcp_inspector=None,
    )

    assert middleware.inspect_tool_requests is False
    assert middleware.inspect_tool_responses is False


def test_constructor_skips_llm_validation_when_model_hooks_disabled():
    settings = AIDefenseSettings(
        llm=EndpointSettings(mode="monitor", endpoint=None, api_key=None, fail_open=True),
        tools=EndpointSettings(mode="off", endpoint=None, api_key=None, fail_open=True),
    )

    middleware = AIDefenseMiddleware(
        settings=settings,
        inspect_model_requests=False,
        inspect_model_responses=False,
        llm_inspector=None,
        mcp_inspector=StubMCPInspector(),
    )

    assert middleware.inspect_model_requests is False
    assert middleware.inspect_model_responses is False


def test_from_env_skips_tool_validation_when_tool_hooks_disabled():
    middleware = AIDefenseMiddleware.from_env(
        env={
            "AI_DEFENSE_API_MODE_LLM_ENDPOINT": "https://example.com",
            "AI_DEFENSE_API_MODE_LLM_API_KEY": "x" * 64,
            "AGENTSEC_API_MODE_LLM": "monitor",
        },
        inspect_tool_requests=False,
        inspect_tool_responses=False,
    )

    assert middleware.inspect_tool_requests is False
    assert middleware.inspect_tool_responses is False


def test_from_env_skips_llm_validation_when_model_hooks_disabled():
    middleware = AIDefenseMiddleware.from_env(
        env={
            "AI_DEFENSE_API_MODE_MCP_ENDPOINT": "https://example.com",
            "AI_DEFENSE_API_MODE_MCP_API_KEY": "y" * 64,
            "AGENTSEC_API_MODE_MCP": "monitor",
        },
        inspect_model_requests=False,
        inspect_model_responses=False,
    )

    assert middleware.inspect_model_requests is False
    assert middleware.inspect_model_responses is False


def test_wrap_tool_call_inspects_response_payload():
    mcp = StubMCPInspector()
    middleware = AIDefenseMiddleware(
        settings=make_settings(llm_mode="off"),
        llm_inspector=StubLLMInspector(),
        mcp_inspector=mcp,
    )

    result = middleware.wrap_tool_call(
        make_tool_request(),
        lambda request: ToolMessage(
            content="page body",
            tool_call_id=request.tool_call["id"],
        ),
    )

    assert isinstance(result, ToolMessage)
    assert [phase for phase, *_ in mcp.calls] == ["request", "response"]


@pytest.mark.asyncio
async def test_awrap_tool_call_supports_command_results():
    mcp = StubMCPInspector()
    middleware = AIDefenseMiddleware(
        settings=make_settings(llm_mode="off"),
        llm_inspector=StubLLMInspector(),
        mcp_inspector=mcp,
    )

    result = await middleware.awrap_tool_call(
        make_tool_request(),
        lambda request: _return_command(request),
    )

    assert isinstance(result, Command)
    assert [phase for phase, *_ in mcp.calls] == ["request", "response"]


async def _return_command(_: ToolCallRequest) -> Command:
    return Command(update={"messages": [AIMessage(content="tool update")]})
