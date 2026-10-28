import asyncio
import os
import sys
import time

sys.path.insert(0, ".")
env = open("C:/Users/Sonny/Documents/Github/AI231-MEX3/.env", encoding="utf-8").read()
key = env.split("OPENROUTER_API_KEY=")[1].splitlines()[0].strip().strip('"')

import httpx

MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
MSG = [{"role": "user", "content": "What is the capital of Australia? One sentence."}]


async def trial(c, label, reasoning):
    t0 = time.monotonic()
    first = None
    try:
        async with c.stream("POST", "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json={"model": MODEL, "messages": MSG, "reasoning": reasoning,
                                  "stream": True}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data:") and '"content"' in line:
                    first = time.monotonic() - t0
                    break
        print(f"{label:22} first content token: {first and round(first*1000)}ms")
    except Exception as e:
        print(f"{label:22} ERROR: {str(e)[:100]}")


async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        await trial(c, "no reasoning param", {})
        await trial(c, "exclude only", {"exclude": True})
        await trial(c, "effort=low+exclude", {"effort": "low", "exclude": True})
        await trial(c, "effort=medium+exclude", {"effort": "medium", "exclude": True})
        await trial(c, "effort=0+exclude", {"effort": 0, "exclude": True})


asyncio.run(main())