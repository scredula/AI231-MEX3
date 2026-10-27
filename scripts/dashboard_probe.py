"""Dashboard UI smoke test: new HTML, /say endpoint, event stream with tool ids."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from agent.observability import EventBus
from agent.dashboard import DashboardServer


async def main() -> None:
    bus = EventBus()
    said = []
    handled = asyncio.Event()

    async def say(text):
        said.append(text)
        handled.set()

    srv = DashboardServer(bus, port=8916, say_handler=say)
    srv.start()
    await asyncio.sleep(2.0)
    import httpx
    c = httpx.AsyncClient()
    r = await c.get("http://127.0.0.1:8916/")
    assert r.status_code == 200 and "Agent activity" in r.content.decode() and "bubble" in r.content.decode()
    print("GET / ok,", len(r.content), "bytes, new UI markers present")

    # guard: the inline JS must be syntactically valid (a broken script silently
    # leaves the UI stuck on 'connecting…')
    import re as _re
    import subprocess
    js = _re.search(r"<script>(.*?)</script>", r.content.decode(), _re.S).group(1)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/ui_check.js").write_text(js, encoding="utf-8")
    chk = subprocess.run(["node", "--check", "logs/ui_check.js"], capture_output=True, text=True)
    assert chk.returncode == 0, f"dashboard JS syntax error:\n{chk.stderr[:400]}"
    print("dashboard JS syntax OK")

    r = await c.post("http://127.0.0.1:8916/say", json={"text": "hello dashboard"})
    print("POST /say:", r.status_code, r.json(), "| handler got:", said)

    # simulate a tool lifecycle and verify stable-id events stream out
    bus.publish({"type": "tool_running", "turn": 1, "id": "call_1", "name": "get_weather",
                 "arguments": '{"location": "Manila"}'})
    bus.publish({"type": "tool_result", "turn": 1, "id": "call_1", "name": "get_weather",
                 "ok": True, "result": "Weather in Manila: 31.1 C, mainly clear."})
    await asyncio.sleep(0.3)
    r = await c.get("http://127.0.0.1:8916/api/events")
    types = [e["type"] for e in r.json()["events"]]
    print("event history types:", types)
    assert types.count("tool_running") == 1 and types.count("tool_result") == 1
    print("DASHBOARD OK")


if __name__ == "__main__":
    asyncio.run(main())