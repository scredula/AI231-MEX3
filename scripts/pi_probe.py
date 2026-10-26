"""Direct PiAdapter probe (debugging aid)."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from coding_agent.adapter import PiAdapter


async def go():
    a = PiAdapter(Config.WORKSPACE_DIR)
    a.set_event_sink(lambda k, d: print(f"EVENT {k} {d[:60]}", flush=True))
    await a.start()
    print("spawned pid", a.proc.pid, flush=True)
    text = await a.run_task("Create a file named hi3.txt containing the word hi, then reply DONE.", timeout=90)
    print("pi replied:", text[:100], flush=True)
    await a.stop()


asyncio.run(go())
print("hi3 exists:", (Config.WORKSPACE_DIR / "hi3.txt").exists(), flush=True)
