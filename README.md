# LangChain AI Defense Middleware

This package turns the Cisco AI Defense `agentsec` LangChain example into a reusable
LangChain `AgentMiddleware`.

Instead of monkeypatching model SDKs with `agentsec.protect()`, this middleware plugs
straight into `langchain.agents.create_agent()` and inspects:

- model requests before the LLM call
- model responses after the LLM call
- tool calls before execution
- tool results after execution

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,examples]"
```

## Environment

The middleware keeps the same environment shape used by the upstream `agentsec`
examples.

```bash
AI_DEFENSE_API_MODE_LLM_ENDPOINT=https://your-runtime-host
AI_DEFENSE_API_MODE_LLM_API_KEY=your-llm-api-key

AI_DEFENSE_API_MODE_MCP_ENDPOINT=https://your-runtime-host
AI_DEFENSE_API_MODE_MCP_API_KEY=your-mcp-api-key

AGENTSEC_API_MODE_LLM=monitor
AGENTSEC_API_MODE_MCP=monitor
AGENTSEC_API_MODE_FAIL_OPEN_LLM=true
AGENTSEC_API_MODE_FAIL_OPEN_MCP=true

OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4.1-mini
```

The endpoint values may be either:

- the runtime host, like `https://us.api.inspect.aidefense.security.cisco.com`
- the older `/api` form, like `https://.../api`
- the full inspection URL, like `https://.../api/v1/inspect/chat`

## Usage

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_aidefense import AIDefenseMiddleware


@tool
def fetch_url(url: str) -> str:
    """Fetch a page."""
    ...


middleware = AIDefenseMiddleware.from_env()

agent = create_agent(
    ChatOpenAI(model="gpt-4.1-mini"),
    tools=[fetch_url],
    middleware=[middleware],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Fetch https://example.com and summarize it",
            }
        ]
    }
)
```

In `monitor` mode the middleware logs policy hits and continues. In `enforce` mode it
raises `SecurityPolicyError`.

## Example

Run the example agent:

```bash
source .venv/bin/activate
python examples/agent.py "Fetch https://example.com and summarize it"
```

## Why This Shape

The upstream example shows the right Cisco AI Defense flow, but it relies on SDK
patching:

- upstream example: [ai-defense-python-sdk/examples/agentsec/2-agent-frameworks/langchain-agent](https://github.com/shiva-guntoju-09/ai-defense-python-sdk/tree/main/examples/agentsec/2-agent-frameworks/langchain-agent)
- LangChain middleware API: [LangChain middleware docs](https://docs.langchain.com/oss/python/langchain/middleware)

This repo keeps the same inspection behavior, but moves it into LangChain's native
middleware hooks so the integration is explicit and testable.
