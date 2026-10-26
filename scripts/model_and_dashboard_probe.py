"""Live test of the switched model: tool calling + time answer, plus dashboard smoke test."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from agent.observability import EventBus
from agent.dashboard import DashboardServer


async def main() -> None:
    print("FAST_MODEL:", Config.FAST_MODEL)
    from services.llm_client import OpenRouterClient
    client = OpenRouterClient()
    tools = [{"type": "function", "function": {"name": "get_time",
              "description": "Get the current local date, time and weekday.",
              "parameters": {"type": "object", "properties": {}, "required": []}}}]
    msg = await client.chat([{"role": "user", "content": "What time is it right now? Use the tool."}],
                            tools=tools)
    calls = msg.get("tool_calls") or []
    print("tool_calls:", json.dumps(calls, ensure_ascii=False)[:200])
    if calls:
        import datetime
        now = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")
        messages = [{"role": "user", "content": "What time is it right now? Use the tool."},
                    msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": now}]
        final = await client.chat(messages, tools=tools)
        print("final answer:", (final.get("content") or "")[:150])

    # dashboard smoke test
    bus = EventBus()
    srv = DashboardServer(bus, port=8915)
    srv.start()
    await asyncio.sleep(2.5)
    import httpx
    r = await httpx.AsyncClient().get("http://127.0.0.1:8915/")
    print("dashboard GET /:", r.status_code, len(r.content), "bytes")
    bus.publish({"type": "turn_start", "turn": 1})
    r2 = await httpx.AsyncClient().get("http://127.0.0.1:8915/api/events")
    print("dashboard /api/events:", r2.status_code, "events:", len(r2.json()["events"]))


if __name__ == "__main__":
    asyncio.run(main())