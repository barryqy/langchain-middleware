from __future__ import annotations

import os
import sys
from urllib.request import Request, urlopen

from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_aidefense import AIDefenseMiddleware

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from langchain_openai import ChatOpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install the example dependencies with: pip install -e '.[examples]'") from exc


if load_dotenv is not None:
    load_dotenv()


@tool
def fetch_url(url: str) -> str:
    """Fetch a URL and return a short text preview."""
    request = Request(
        url,
        headers={"User-Agent": "langchain-aidefense-middleware/0.1.0"},
    )
    with urlopen(request, timeout=10) as response:
        body = response.read(4000).decode("utf-8", errors="replace")
    return body


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Fetch https://example.com and summarize it."

    agent = create_agent(
        ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini")),
        tools=[fetch_url],
        middleware=[AIDefenseMiddleware.from_env()],
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }
    )

    for message in result["messages"]:
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "text", None) or getattr(message, "content", "")
        print(f"{role}: {content}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

