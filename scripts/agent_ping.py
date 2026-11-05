import asyncio
import sys

sys.path.insert(0, ".")


async def main():
    from agent.main import _make_coordinator
    from agent.conversation import ConversationAgent
    coordinator = _make_coordinator()
    coordinator.on_announce(lambda t: print(f"[announce] {t}"))
    agent = ConversationAgent(coordinator)
    t0 = asyncio.get_event_loop().time()
    try:
        reply = await asyncio.wait_for(agent.reply("What time is it in New York?"), timeout=90)
        print(f"reply in {asyncio.get_event_loop().time()-t0:.1f}s:", reply[:200])
    except asyncio.TimeoutError:
        print(f"HUNG after {asyncio.get_event_loop().time()-t0:.1f}s")
        import faulthandler
        faulthandler.dump_traceback()


asyncio.run(main())