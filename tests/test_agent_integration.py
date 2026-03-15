from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

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
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def inspect_conversation(self, messages, metadata):
        phase = "response" if messages and messages[0]["role"] == "assistant" else "request"
        self.calls.append((phase, messages))
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
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def inspect_request(self, tool_name, arguments, metadata, method="tools/call"):
        self.calls.append(("request", tool_name, arguments))
        return self.request_decision

    def inspect_response(self, tool_name, arguments, result, metadata, method="tools/call"):
        self.calls.append(("response", tool_name, arguments))
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
        llm=EndpointSettings(mode=llm_mode, endpoint="https://example.com", api_key="x" * 64, fail_open=True),
        tools=EndpointSettings(mode=tool_mode, endpoint="https://example.com", api_key="x" * 64, fail_open=True),
        violation_behavior=violation_behavior,
        violation_message=violation_message,
    )


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL."""
    return f"fetched {url}"


class ToolFriendlyFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def test_create_agent_runs_with_model_middleware():
    llm = StubLLMInspector()
    middleware = AIDefenseMiddleware(
        settings=make_settings(tool_mode="off"),
        llm_inspector=llm,
        mcp_inspector=StubMCPInspector(),
    )
    agent = create_agent(model=FakeListChatModel(responses=["hello"]), middleware=[middleware])

    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})

    assert result["messages"][-1].content == "hello"
    assert [phase for phase, _ in llm.calls] == ["request", "response"]


def test_create_agent_runs_with_tool_middleware():
    llm = StubLLMInspector()
    mcp = StubMCPInspector()
    middleware = AIDefenseMiddleware(
        settings=make_settings(),
        llm_inspector=llm,
        mcp_inspector=mcp,
    )
    agent = create_agent(
        model=ToolFriendlyFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "fetch_url",
                            "args": {"url": "https://example.com"},
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        tools=[fetch_url],
        middleware=[middleware],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    assert result["messages"][-1].content == "done"
    assert [phase for phase, _, _ in mcp.calls] == ["request", "response"]
    assert len(llm.calls) == 4


def test_create_agent_blocks_in_enforce_mode():
    middleware = AIDefenseMiddleware(
        settings=make_settings(llm_mode="enforce", tool_mode="off"),
        llm_inspector=StubLLMInspector(request_decision=block_decision("bad prompt")),
        mcp_inspector=StubMCPInspector(),
    )
    agent = create_agent(model=FakeListChatModel(responses=["hello"]), middleware=[middleware])

    with pytest.raises(SecurityPolicyError, match="llm_request"):
        agent.invoke({"messages": [{"role": "user", "content": "hi"}]})


def test_create_agent_ends_on_model_violation():
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="enforce",
            tool_mode="off",
            violation_behavior="end",
        ),
        llm_inspector=StubLLMInspector(request_decision=block_decision("bad prompt")),
        mcp_inspector=StubMCPInspector(),
    )
    agent = create_agent(model=FakeListChatModel(responses=["hello"]), middleware=[middleware])

    result = agent.invoke({"messages": [{"role": "user", "content": "hi"}]})

    assert result["messages"][-1].content == "AI Defense blocked llm_request: bad prompt"


def test_create_agent_ends_after_tool_violation():
    middleware = AIDefenseMiddleware(
        settings=make_settings(
            llm_mode="monitor",
            tool_mode="enforce",
            violation_behavior="end",
        ),
        llm_inspector=StubLLMInspector(),
        mcp_inspector=StubMCPInspector(request_decision=block_decision("unsafe tool")),
    )
    agent = create_agent(
        model=ToolFriendlyFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "fetch_url",
                            "args": {"url": "https://example.com"},
                        }
                    ],
                ),
                AIMessage(content="should not happen"),
            ]
        ),
        tools=[fetch_url],
        middleware=[middleware],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    assert result["messages"][-1].content == "AI Defense blocked tool_request: unsafe tool"
    assert result["messages"][-2].content == "AI Defense blocked tool_request: unsafe tool"
