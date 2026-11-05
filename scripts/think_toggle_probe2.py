import asyncio, json, sys, time
sys.path.insert(0, ".")
env = open("C:/Users/Sonny/Documents/Github/AI231-MEX3/.env", encoding="utf-8").read()
key = env.split("OPENROUTER_API_KEY=")[1].splitlines()[0].strip().strip('"')
import httpx

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
TOOLS = [{"type": "function", "function": {
    "name": "get_time", "description": "Get the current date/time.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}}}}]

async def run(c, label, reasoning, with_tool_result):
    msgs = [
        {"role": "system", "content": "Answer in one short sentence."},
        {"role": "user", "content": "What time is it in New York?"},
    ]
    if with_tool_result:
        msgs += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call1", "type": "function",
                 "function": {"name": "get_time", "arguments": "{\"timezone\": \"America/New_York\"}"}}]},
            {"role": "tool", "tool_call_id": "call1",
             "content": "Monday, 04 November 2026, 00:14:54 (EST, UTC-0500)"},
        ]
    body = {"model": MODEL, "stream": True, "messages": msgs, "tools": TOOLS, "reasoning": reasoning}
    t0 = time.monotonic()
    content = ""
    tool_calls = []
    async with c.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"}, json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:].strip())
            except Exception:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            for tc in delta.get("tool_calls") or []:
                tool_calls.append(tc.get("function", {}).get("name"))
    dt = time.monotonic() - t0
    print(f"--- {label} ({dt:.1f}s) ---")
    print(f"tool_calls={tool_calls or 'none'} content={content[:200]!r}")

async def main():
    async with httpx.AsyncClient(timeout=180) as c:
        await run(c, "ROUND2 exclude only", {"exclude": True}, True)
        await run(c, "ROUND2 enabled=false", {"enabled": False, "exclude": True}, True)

asyncio.run(main())
