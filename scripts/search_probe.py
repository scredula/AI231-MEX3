"""Live test of the search_web tool through the real LLM (model decides to call it)."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> None:
    from services.llm_client import OpenRouterClient
    from services.websearch import search_web

    tools = [{"type": "function", "function": {"name": "search_web",
              "description": "Search the web for current information (news, weather, facts).",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                             "required": ["query"]}}}]
    client = OpenRouterClient()
    messages = [{"role": "user", "content": "What's the weather like in Manila right now? Search the web."}]
    for _ in range(3):
        msg = await client.chat(messages, tools=tools)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            print("answer:", (msg.get("content") or "")[:300])
            return
        for call in calls:
            q = json.loads(call["function"]["arguments"]).get("query", "")
            print("searching:", q)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": search_web(q)})
    print("loop limit hit")


if __name__ == "__main__":
    asyncio.run(main())