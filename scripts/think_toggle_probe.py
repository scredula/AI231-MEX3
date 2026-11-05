import asyncio
import sys
import time

sys.path.insert(0, ".")
env = open("C:/Users/Sonny/Documents/Github/AI231-MEX3/.env", encoding="utf-8").read()
key = env.split("OPENROUTER_API_KEY=")[1].splitlines()[0].strip().strip('"')

import httpx

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
TOOLS = [{"type": "function", "function": {
    "name": "get_time", "description": "Get the current date/time.",
    "parameters": {"type": "object", "properties": {}}}}]


async def trial(c, label, reasoning, with_tools=False):
    body = {
        "model": MODEL, "stream": True,
        "messages": ([{"role": "system", "content": "Answer in one short sentence."}]
                     + [{"role": "user", "content": "What time is it in New York?"}]),
        "reasoning": reasoning,
    }
    if with_tools:
        body["tools"] = TOOLS
    t0 = time.monotonic()
    content, saw_reasoning = "", False
    try:
        async with c.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}"}, json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                if line == "data: [DONE]":
                    break
                import json
                try:
                    d = json.loads(line[5:].strip())
                except Exception:
                    continue
                delta = (d.get("choices") or [{}])[0].get("delta", {})
                if delta.get("reasoning"):
                    saw_reasoning = True
                if delta.get("content"):
                    content += delta["content"]
        dt = time.monotonic() - t0
        print(f"--- {label} ({dt:.1f}s, reasoning_deltas={saw_reasoning}) ---")
        print("content:", repr(content[:220]))
    except Exception as e:
        print(f"--- {label} ERROR: {str(e)[:120]}")


async def main():
    async with httpx.AsyncClient(timeout=180) as c:
        await trial(c, "plain, exclude only", {"exclude": True})
        await trial(c, "plain, enabled=false", {"enabled": False, "exclude": True})
        await trial(c, "TOOLS + exclude only", {"exclude": True}, with_tools=True)
        await trial(c, "TOOLS + enabled=false", {"enabled": False, "exclude": True}, with_tools=True)


asyncio.run(main())