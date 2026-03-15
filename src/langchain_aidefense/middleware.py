from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, TypeAlias

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from .config import AIDefenseSettings
from .decision import Decision
from .exceptions import SecurityPolicyError
from .inspectors import LLMInspector, MCPInspector


MetadataFactory: TypeAlias = Callable[[str, dict[str, Any]], Mapping[str, Any] | None]
ToolMethodResolver: TypeAlias = Callable[[ToolCallRequest], str]


def _default_tool_method(_: ToolCallRequest) -> str:
    return "tools/call"


class AIDefenseMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        settings: AIDefenseSettings,
        llm_inspector: Any | None = None,
        mcp_inspector: Any | None = None,
        metadata_factory: MetadataFactory | None = None,
        tool_method_resolver: ToolMethodResolver | None = None,
        inspect_model_requests: bool = True,
        inspect_model_responses: bool = True,
        inspect_tool_requests: bool = True,
        inspect_tool_responses: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.metadata_factory = metadata_factory
        self.tool_method_resolver = tool_method_resolver or _default_tool_method
        self.inspect_model_requests = inspect_model_requests
        self.inspect_model_responses = inspect_model_responses
        self.inspect_tool_requests = inspect_tool_requests
        self.inspect_tool_responses = inspect_tool_responses
        self.logger = logger or logging.getLogger("langchain_aidefense.middleware")
        self.tools = []

        if llm_inspector is None:
            self.settings.llm.validate("LLM")
        if mcp_inspector is None:
            self.settings.tools.validate("tool")

        self._llm_inspector = llm_inspector or self._make_llm_inspector()
        self._mcp_inspector = mcp_inspector or self._make_mcp_inspector()

    @classmethod
    def from_env(cls, **kwargs: Any) -> "AIDefenseMiddleware":
        return cls(settings=AIDefenseSettings.from_env(), **kwargs)

    def _make_llm_inspector(self) -> LLMInspector:
        return LLMInspector(
            api_key=self.settings.llm.api_key,
            endpoint=self.settings.llm.endpoint,
            default_rules=list(self.settings.llm_default_rules) or None,
            entity_types=list(self.settings.llm_entity_types) or None,
            timeout_ms=self.settings.timeout_ms,
            retry_total=self.settings.retry_total,
            retry_backoff=self.settings.retry_backoff,
            fail_open=self.settings.llm.fail_open,
            logger_instance=self.logger,
        )

    def _make_mcp_inspector(self) -> MCPInspector:
        return MCPInspector(
            api_key=self.settings.tools.api_key,
            endpoint=self.settings.tools.endpoint,
            timeout_ms=self.settings.timeout_ms,
            retry_total=self.settings.retry_total,
            retry_backoff=self.settings.retry_backoff,
            fail_open=self.settings.tools.fail_open,
            logger_instance=self.logger,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        if self.settings.llm.enabled and self.inspect_model_requests:
            self._inspect_model_messages(
                self._request_messages(request),
                phase="llm_request",
                mode=self.settings.llm.mode,
                details={"request": request},
            )

        response = handler(request)

        if self.settings.llm.enabled and self.inspect_model_responses:
            self._inspect_model_messages(
                self._response_messages(response),
                phase="llm_response",
                mode=self.settings.llm.mode,
                details={"request": request, "response": response},
            )

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelResponse | AIMessage | ExtendedModelResponse:
        if self.settings.llm.enabled and self.inspect_model_requests:
            await self._ainspect_model_messages(
                self._request_messages(request),
                phase="llm_request",
                mode=self.settings.llm.mode,
                details={"request": request},
            )

        response = await handler(request)

        if self.settings.llm.enabled and self.inspect_model_responses:
            await self._ainspect_model_messages(
                self._response_messages(response),
                phase="llm_response",
                mode=self.settings.llm.mode,
                details={"request": request, "response": response},
            )

        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = _tool_name(request)
        tool_args = dict(request.tool_call.get("args") or {})
        method = self.tool_method_resolver(request)
        metadata = self._metadata(
            "tool_request",
            tool_name=tool_name,
            tool_args=tool_args,
            method=method,
            request=request,
        )

        if self.settings.tools.enabled and self.inspect_tool_requests:
            decision = self._mcp_inspector.inspect_request(
                tool_name=tool_name,
                arguments=tool_args,
                metadata=metadata,
                method=method,
            )
            self._handle_decision(decision, phase="tool_request", mode=self.settings.tools.mode)

        result = handler(request)

        if self.settings.tools.enabled and self.inspect_tool_responses:
            decision = self._mcp_inspector.inspect_response(
                tool_name=tool_name,
                arguments=tool_args,
                result=_tool_result_payload(result),
                metadata=self._metadata(
                    "tool_response",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    method=method,
                    result=result,
                    request=request,
                ),
                method=method,
            )
            self._handle_decision(decision, phase="tool_response", mode=self.settings.tools.mode)

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command[Any]:
        tool_name = _tool_name(request)
        tool_args = dict(request.tool_call.get("args") or {})
        method = self.tool_method_resolver(request)
        metadata = self._metadata(
            "tool_request",
            tool_name=tool_name,
            tool_args=tool_args,
            method=method,
            request=request,
        )

        if self.settings.tools.enabled and self.inspect_tool_requests:
            decision = await self._mcp_inspector.ainspect_request(
                tool_name=tool_name,
                arguments=tool_args,
                metadata=metadata,
                method=method,
            )
            self._handle_decision(decision, phase="tool_request", mode=self.settings.tools.mode)

        result = await handler(request)

        if self.settings.tools.enabled and self.inspect_tool_responses:
            decision = await self._mcp_inspector.ainspect_response(
                tool_name=tool_name,
                arguments=tool_args,
                result=_tool_result_payload(result),
                metadata=self._metadata(
                    "tool_response",
                    tool_name=tool_name,
                    tool_args=tool_args,
                    method=method,
                    result=result,
                    request=request,
                ),
                method=method,
            )
            self._handle_decision(decision, phase="tool_response", mode=self.settings.tools.mode)

        return result

    def close(self) -> None:
        self._llm_inspector.close()
        self._mcp_inspector.close()

    async def aclose(self) -> None:
        await self._llm_inspector.aclose()
        await self._mcp_inspector.aclose()

    def _inspect_model_messages(
        self,
        messages: list[dict[str, str]],
        *,
        phase: str,
        mode: str,
        details: dict[str, Any],
    ) -> None:
        if not messages:
            return

        decision = self._llm_inspector.inspect_conversation(
            messages=messages,
            metadata=self._metadata(phase, **details),
        )
        self._handle_decision(decision, phase=phase, mode=mode)

    async def _ainspect_model_messages(
        self,
        messages: list[dict[str, str]],
        *,
        phase: str,
        mode: str,
        details: dict[str, Any],
    ) -> None:
        if not messages:
            return

        decision = await self._llm_inspector.ainspect_conversation(
            messages=messages,
            metadata=self._metadata(phase, **details),
        )
        self._handle_decision(decision, phase=phase, mode=mode)

    def _handle_decision(self, decision: Decision, *, phase: str, mode: str) -> None:
        if decision.allows():
            return

        summary = _decision_summary(decision)
        if mode == "enforce":
            raise SecurityPolicyError(decision, f"AI Defense blocked {phase}: {summary}")

        self.logger.warning("AI Defense flagged %s in %s mode: %s", phase, mode, summary)

    def _metadata(self, phase: str, **details: Any) -> dict[str, Any]:
        data = {
            "src_app": "langchain",
            "user_agent": "langchain-aidefense-middleware/0.1.0",
        }

        request_obj = details.get("request")
        model = getattr(request_obj, "model", None) if request_obj is not None else None
        if model is not None:
            data["dst_app"] = _model_name(model)

        if details.get("tool_name"):
            data["dst_app"] = str(details["tool_name"])

        if self.metadata_factory is None:
            return data

        extra = self.metadata_factory(phase, details)
        if not extra:
            return data

        for key, value in dict(extra).items():
            if value is not None:
                data[key] = value

        return data

    def _request_messages(self, request: ModelRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if request.system_message is not None:
            msg = _inspection_message(request.system_message)
            if msg is not None:
                messages.append(msg)

        for message in request.messages:
            tmp = _inspection_message(message)
            if tmp is not None:
                messages.append(tmp)

        return messages

    def _response_messages(
        self,
        response: ModelResponse | AIMessage | ExtendedModelResponse,
    ) -> list[dict[str, str]]:
        if isinstance(response, ExtendedModelResponse):
            raw_messages = response.model_response.result
        elif isinstance(response, AIMessage):
            raw_messages = [response]
        else:
            raw_messages = response.result

        messages: list[dict[str, str]] = []
        for message in raw_messages:
            if not isinstance(message, AIMessage):
                continue
            tmp = _inspection_message(message)
            if tmp is not None:
                messages.append(tmp)

        return messages


AgentSecMiddleware = AIDefenseMiddleware


def _inspection_message(message: BaseMessage) -> dict[str, str] | None:
    content = _message_text(message)
    if not content:
        return None

    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "assistant"
        content = f"Tool result: {content}"
    else:
        role = "user"

    return {"role": role, "content": content}


def _message_text(message: BaseMessage) -> str:
    text = getattr(message, "text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()

    if isinstance(message, AIMessage) and message.tool_calls:
        return json.dumps({"tool_calls": message.tool_calls}, default=_json_default, sort_keys=True)

    raw = getattr(message, "content", None)
    return _content_text(raw)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                else:
                    parts.append(json.dumps(item, default=_json_default, sort_keys=True))
                continue

            parts.append(str(item))
        return "\n".join(part for part in parts if part)

    return str(content).strip()


def _tool_result_payload(result: ToolMessage | Command[Any]) -> Any:
    if isinstance(result, ToolMessage):
        payload: dict[str, Any] = {
            "content": _tool_content_blocks(result.content),
        }
        status = getattr(result, "status", None)
        if status:
            payload["status"] = status
        return payload

    update = getattr(result, "update", None)
    if update is None:
        return {"content": [{"type": "text", "text": repr(result)}]}

    if is_dataclass(update):
        return asdict(update)

    return update


def _tool_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                blocks.append(item)
            else:
                blocks.append({"type": "text", "text": str(item)})
        return blocks

    if isinstance(content, dict):
        return [content]

    return [{"type": "text", "text": _content_text(content)}]


def _tool_name(request: ToolCallRequest) -> str:
    if request.tool is not None:
        return request.tool.name
    return str(request.tool_call.get("name", "unknown_tool"))


def _model_name(model: Any) -> str:
    for attr in ("model_name", "model", "deployment_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return model.__class__.__name__


def _decision_summary(decision: Decision) -> str:
    if decision.explanation:
        return decision.explanation
    if decision.reasons:
        return "; ".join(str(item) for item in decision.reasons)
    return "policy violation"


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return str(value)
