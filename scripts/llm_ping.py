import asyncio
import sys

sys.path.insert(0, ".")


async def main():
    from services.llm_client import OpenRouterClient
    c = OpenRouterClient()
    t0 = asyncio.get_event_loop().time()
    msg = await asyncio.wait_for(
        c.chat([{"role": "user", "content": "Reply with exactly: PONG"}]),
        timeout=60)
    dt = asyncio.get_event_loop().time() - t0
    print(f"chat OK in {dt:.1f}s:", repr(str(msg.get("content"))[:120]))


asyncio.run(main())