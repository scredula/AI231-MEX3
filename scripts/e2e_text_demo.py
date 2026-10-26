"""End-to-end text-mode demo: chat -> background website build -> verify files exist.

  .venv/Scripts/python.exe scripts/e2e_text_demo.py "Build me a small portfolio website"
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from agent.main import _make_coordinator
from agent.conversation import ConversationAgent
from agent.task_events import TaskState


async def main() -> None:
    user_prompt = " ".join(sys.argv[1:]) or "Build me a small portfolio website for a student named Alex."
    coordinator = _make_coordinator()
    coordinator.on_announce(lambda t: print(f"[announce] {t}"))
    agent = ConversationAgent(coordinator)

    reply = await agent.reply(user_prompt)
    print("agent>", reply)

    # wait for the background task to finish
    task_ids = [t.id for t in coordinator.store.list()]
    if not task_ids:
        print("no background task was submitted")
        return
    for tid in task_ids:
        t = await coordinator.status(tid)
        for _ in range(240):
            t = await coordinator.status(tid)
            if t.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
                break
            await asyncio.sleep(2)
        print(f"task {tid}: {t.state.value} -> {t.result[:300]}")

    site = Config.WORKSPACE_DIR / "site"
    files = sorted(p.name for p in site.rglob("*") if p.is_file()) if site.exists() else []
    print("site files:", files)
    assert "index.html" in files, "index.html missing — build failed"
    print("E2E OK")


if __name__ == "__main__":
    asyncio.run(main())
