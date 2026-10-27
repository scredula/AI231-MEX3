import asyncio
import os
import sys

sys.path.insert(0, ".")
env = open("C:/Users/Sonny/Documents/Github/AI231-MEX3/.env", encoding="utf-8").read()
key = env.split("OPENROUTER_API_KEY=")[1].splitlines()[0].strip().strip('"')

import httpx


async def main():
    base = {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "messages": [{"role": "user", "content": "What is 2+2? Answer in one short sentence."}],
    }
    async with httpx.AsyncClient(timeout=90) as c:
        for label, extra in (("no reasoning param", {}),
                             ("reasoning exclude", {"reasoning": {"exclude": True}})):
            body = dict(base, **extra)
            r = await c.post("https://openrouter.ai/api/v1/chat/completions",
                             headers={"Authorization": f"Bearer {key}"}, json=body)
            d = r.json()
            m = (d.get("choices") or [{}])[0].get("message", {})
            print(f"--- {label} ---")
            print("content:", repr((m.get("content") or "")[:100]))
            print("reasoning field present:", bool(m.get("reasoning")))


asyncio.run(main())