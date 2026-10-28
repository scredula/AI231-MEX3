"""Live E2E: exercise the tool + announcement path via /say."""
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> None:
    c = httpx.AsyncClient()
    print("GET /:", (await c.get("http://127.0.0.1:8900/")).status_code)
    r = await c.post("http://127.0.0.1:8900/say",
                     json={"text": "Search the web for today's biggest AI news headlines and summarize them."})
    print("say:", r.status_code, r.text[:60])
    await asyncio.sleep(45)
    r = await c.get("http://127.0.0.1:8900/api/events")
    evs = r.json()["events"]
    print(len(evs), "events:", dict(Counter(e["type"] for e in evs)))
    for e in evs:
        if e["type"] == "announcement":
            print("ANNOUNCEMENT:", repr(e["text"]))
        if e["type"] == "tool_result":
            print("TOOL:", e.get("name"), "ok:", e.get("ok"), "duration:", e.get("duration"))
        if e["type"] == "bot_speech_start":
            print("LATENCIES:", {k: v for k, v in e["latencies"].items() if v is not None})
        if e["type"] == "error":
            print("ERROR:", e.get("error", "")[:120])


if __name__ == "__main__":
    asyncio.run(main())