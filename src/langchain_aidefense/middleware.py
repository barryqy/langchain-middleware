from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Annotated, Any, Callable, Mapping, TypeAlias

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    AgentState,
    EphemeralValue,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from typing_extensions import TypedDict

from .config import AIDefenseSettings, _normalize_violation_behavior
from .decision import Decision
from .exceptions import SecurityPolicyError
from .inspectors import LLMInspector, MCPInspector


MetadataFactory: TypeAlias = Callable[[str, dict[str, Any]], Mapping[str, Any] | None]
ToolMethodResolver: TypeAlias = Callable[[ToolCallRequest], str]
ModelCallResult: TypeAlias = ModelResponse | AIMessage | ExtendedModelResponse
ToolCallResult: TypeAlias = ToolMessage | Command[Any]

DEFAULT_VIOLATION_TEMPLATE = "AI Defense blocked {phase}: {summary}"
_PENDING_END_MESSAGE_KEY = "aidefense_pending_end_message"


class AIDefenseState(AgentState[Any], total=False):
    aidefense_pending_end_message: Annotated[str | None, EphemeralValue, PrivateStateAttr]


def _default_tool_method(_: ToolCallRequest) -> str:
    return "tools/call"


class AIDefenseMiddleware(AgentMiddleware):
    state_schema = AIDefenseState

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
        violation_behavior: str | None = None,
        violation_message: str | None = None,
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
        self.violation_behavior = _normalize_violation_behavior(
            violation_behavior or self.settings.violation_behavior
        )
        self.violation_message = (
            self.settings.violation_message if violation_message is None else violation_message
        )
        self.logger = logger or logging.getLogger("langchain_aidefense.middleware")
        self.tools = []
        self._model_inspection_active = self.inspect_model_requests or self.inspect_model_responses
        self._tool_inspection_active = self.inspect_tool_requests or self.inspect_tool_responses

        if llm_inspector is None and self.settings.llm.enabled and self._model_inspection_active:
            self.settings.llm.validate("LLM")
        if mcp_inspector is None and self.settings.tools.enabled and self._tool_inspection_active:
            self.settings.tools.validate("tool")

        self._llm_inspector = llm_inspector or self._make_llm_inspector()
        self._mcp_inspector = mcp_inspector or self._make_mcp_inspector()

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> "AIDefenseMiddleware":
        return cls(settings=AIDefenseSettings.from_env(env=env, validate=False), **kwargs)

    def _make_llm_inspector(self) -> LLMInspector:
        return LLMInspector(
            api_key=self.settings.llm.api_key,
            endpoint=self.settings.llm.endpoint,
            default_rules=list(self.settings.llm_default_rules) or None,
            entity_types=list(self.settings.llm_entity_types) or None,
            timeout_ms=self.settings.timeout_ms,
            retry_total=self.settings.retry_total,
            retry_backoff=self.settings.retry_backoff,
            retry_status_codes=list(self.settings.retry_status_codes) or None,
            pool_max_connections=self.settings.pool_max_connections,
            pool_max_keepalive=self.settings.pool_max_keepalive,
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
            retry_status_codes=list(self.settings.retry_status_codes) or None,
            pool_max_connections=self.settings.pool_max_connections,
            pool_max_keepalive=self.settings.pool_max_keepalive,
            fail_open=self.settings.tools.fail_open,
            logger_instance=self.logger,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        current_request = request

        if self.settings.llm.enabled and self.inspect_model_requests:
            current_request, blocked = self._inspect_model_request(current_request)
            if blocked is not None:
                return blocked

        response = handler(current_request)

        if self.settings.llm.enabled and self.inspect_model_responses:
            blocked = self._inspect_model_response(current_request, response)
            if blocked is not None:
                return blocked

        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> ModelCallResult:
        current_request = request

        if self.settings.llm.enabled and self.inspect_model_requests:
            current_request, blocked = await self._ainspect_model_request(current_request)
            if blocked is not None:
                return blocked

        response = await handler(current_request)

        if self.settings.llm.enabled and self.inspect_model_responses:
            blocked = await self._ainspect_model_response(current_request, response)
            if blocked is not None:
                return blocked

        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolCallResult:
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
            blocked = self._resolve_tool_violation(
                request,
                decision,
                phase="tool_request",
                mode=self.settings.tools.mode,
            )
            if blocked is not None:
                return blocked

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
            blocked = self._resolve_tool_violation(
                request,
                decision,
                phase="tool_response",
                mode=self.settings.tools.mode,
            )
            if blocked is not None:
                return blocked

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolCallResult:
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
            blocked = self._resolve_tool_violation(
                request,
                decision,
                phase="tool_request",
                mode=self.settings.tools.mode,
            )
            if blocked is not None:
                return blocked

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
            blocked = self._resolve_tool_violation(
                request,
                decision,
                phase="tool_response",
                mode=self.settings.tools.mode,
            )
            if blocked is not None:
                return blocked

        return result

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: AIDefenseState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        return self._consume_pending_end(state)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AIDefenseState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        del runtime
        return self._consume_pending_end(state)

    def close(self) -> None:
        self._llm_inspector.close()
        self._mcp_inspector.close()

    async def aclose(self) -> None:
        await self._llm_inspector.aclose()
        await self._mcp_inspector.aclose()

    def _inspect_model_request(
        self,
        request: ModelRequest,
    ) -> tuple[ModelRequest, ModelCallResult | None]:
        messages = self._request_messages(request)
        if not messages:
            return request, None

        decision = self._llm_inspector.inspect_conversation(
            messages=messages,
            metadata=self._metadata("llm_request", request=request),
        )
        return self._resolve_model_request_violation(
            request,
            decision,
            phase="llm_request",
            mode=self.settings.llm.mode,
        )

    async def _ainspect_model_request(
        self,
        request: ModelRequest,
    ) -> tuple[ModelRequest, ModelCallResult | None]:
        messages = self._request_messages(request)
        if not messages:
            return request, None

        decision = await self._llm_inspector.ainspect_conversation(
            messages=messages,
            metadata=self._metadata("llm_request", request=request),
        )
        return self._resolve_model_request_violation(
            request,
            decision,
            phase="llm_request",
            mode=self.settings.llm.mode,
        )

    def _inspect_model_response(
        self,
        request: ModelRequest,
        response: ModelCallResult,
    ) -> ModelCallResult | None:
        messages = self._response_messages(response)
        if not messages:
            return None

        decision = self._llm_inspector.inspect_conversation(
            messages=messages,
            metadata=self._metadata("llm_response", request=request, response=response),
        )
        return self._resolve_model_response_violation(
            response,
            decision,
            phase="llm_response",
            mode=self.settings.llm.mode,
        )

    async def _ainspect_model_response(
        self,
        request: ModelRequest,
        response: ModelCallResult,
    ) -> ModelCallResult | None:
        messages = self._response_messages(response)
        if not messages:
            return None

        decision = await self._llm_inspector.ainspect_conversation(
            messages=messages,
            metadata=self._metadata("llm_response", request=request, response=response),
        )
        return self._resolve_model_response_violation(
            response,
            decision,
            phase="llm_response",
            mode=self.settings.llm.mode,
        )

    def _resolve_model_request_violation(
        self,
        request: ModelRequest,
        decision: Decision,
        *,
        phase: str,
        mode: str,
    ) -> tuple[ModelRequest, ModelCallResult | None]:
        if decision.allows():
            return request, None

        if mode != "enforce":
            self._log_violation(decision, phase=phase, mode=mode)
            return request, None

        violation_text = self._violation_text(decision, phase=phase, mode=mode)
        if self.violation_behavior == "error":
            self._raise_violation(decision, message=violation_text)
        if self.violation_behavior == "end":
            return request, AIMessage(content=violation_text)
        return self._replace_model_request(request, violation_text), None

    def _resolve_model_response_violation(
        self,
        response: ModelCallResult,
        decision: Decision,
        *,
        phase: str,
        mode: str,
    ) -> ModelCallResult | None:
        if decision.allows():
            return None

        if mode != "enforce":
            self._log_violation(decision, phase=phase, mode=mode)
            return None

        violation_text = self._violation_text(decision, phase=phase, mode=mode)
        if self.violation_behavior == "error":
            self._raise_violation(decision, message=violation_text)
        return _replace_model_call_result(response, violation_text)

    def _resolve_tool_violation(
        self,
        request: ToolCallRequest,
        decision: Decision,
        *,
        phase: str,
        mode: str,
    ) -> ToolCallResult | None:
        if decision.allows():
            return None

        if mode != "enforce":
            self._log_violation(decision, phase=phase, mode=mode)
            return None

        violation_text = self._violation_text(decision, phase=phase, mode=mode)
        if self.violation_behavior == "error":
            self._raise_violation(decision, message=violation_text)

        blocked_message = self._blocked_tool_message(request, violation_text)
        if self.violation_behavior == "replace":
            return blocked_message
        return self._end_after_tool_violation(blocked_message, violation_text)

    def _consume_pending_end(self, state: AIDefenseState) -> dict[str, Any] | None:
        pending_message = state.get(_PENDING_END_MESSAGE_KEY)
        if not pending_message:
            return None

        return {
            "jump_to": "end",
            _PENDING_END_MESSAGE_KEY: None,
        }

    def _replace_model_request(self, request: ModelRequest, text: str) -> ModelRequest:
        overrides: dict[str, Any] = {}
        replaced_messages = _replace_message_list(request.messages, text)
        if replaced_messages != list(request.messages):
            overrides["messages"] = replaced_messages

        if request.system_message is not None:
            overrides["system_message"] = _replace_message(request.system_message, text)

        new_state = _replace_state_messages(request.state, text)
        if new_state is not request.state:
            overrides["state"] = new_state

        if not overrides:
            return request
        return request.override(**overrides)

    def _blocked_tool_message(self, request: ToolCallRequest, text: str) -> ToolMessage:
        return ToolMessage(
            content=text,
            name=_tool_name(request),
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def _end_after_tool_violation(
        self,
        blocked_message: ToolMessage,
        violation_text: str,
    ) -> ToolCallResult:
        return Command(
            update={
                "messages": [blocked_message, AIMessage(content=violation_text)],
                _PENDING_END_MESSAGE_KEY: violation_text,
            }
        )

    def _violation_text(self, decision: Decision, *, phase: str, mode: str) -> str:
        template = self.violation_message or DEFAULT_VIOLATION_TEMPLATE
        reasons = "; ".join(str(item) for item in decision.reasons) or "policy violation"
        values = {
            "action": decision.action,
            "classifications": ", ".join(decision.classifications or ()),
            "event_id": decision.event_id or "",
            "explanation": decision.explanation or reasons,
            "mode": mode,
            "phase": phase,
            "reasons": reasons,
            "severity": decision.severity or "",
            "summary": _decision_summary(decision),
        }

        try:
            return template.format(**values)
        except KeyError:
            return template

    def _raise_violation(self, decision: Decision, *, message: str) -> None:
        raise SecurityPolicyError(decision, message)

    def _log_violation(self, decision: Decision, *, phase: str, mode: str) -> None:
        self.logger.warning(
            "AI Defense flagged %s in %s mode: %s",
            phase,
            mode,
            _decision_summary(decision),
        )

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


def _replace_model_call_result(response: ModelCallResult, text: str) -> ModelCallResult:
    if isinstance(response, ExtendedModelResponse):
        return ExtendedModelResponse(
            model_response=_replace_model_response(response.model_response, text),
            command=response.command,
        )

    if isinstance(response, AIMessage):
        return _replace_message(response, text)

    return _replace_model_response(response, text)


def _replace_model_response(response: ModelResponse, text: str) -> ModelResponse:
    return ModelResponse(
        result=[
            _replace_message(message, text) if isinstance(message, AIMessage) else message
            for message in response.result
        ],
        structured_response=response.structured_response,
    )


def _replace_state_messages(state: Any, text: str) -> Any:
    if not isinstance(state, dict):
        return state

    raw_messages = state.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return state

    updated_messages: list[Any] = []
    changed = False
    for item in raw_messages:
        if isinstance(item, BaseMessage):
            updated_messages.append(_replace_message(item, text))
            changed = True
            continue

        if isinstance(item, dict):
            updated_item = dict(item)
            updated_item["content"] = text
            if updated_item.get("role") == "assistant":
                updated_item["tool_calls"] = []
                updated_item["invalid_tool_calls"] = []
            if updated_item.get("type") == "tool":
                updated_item["status"] = "error"
            updated_messages.append(updated_item)
            changed = True
            continue

        updated_messages.append(item)

    if not changed:
        return state

    new_state = dict(state)
    new_state["messages"] = updated_messages
    return new_state


def _replace_message_list(messages: list[BaseMessage], text: str) -> list[BaseMessage]:
    return [_replace_message(message, text) for message in messages]


def _replace_message(message: BaseMessage, text: str) -> BaseMessage:
    updates: dict[str, Any] = {"content": text}

    if isinstance(message, AIMessage):
        updates["tool_calls"] = []
        updates["invalid_tool_calls"] = []

    if isinstance(message, ToolMessage):
        updates["status"] = "error"

    return message.model_copy(update=updates)


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
