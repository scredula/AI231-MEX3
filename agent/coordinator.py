"""Task coordinator.

Connects the conversational agent to the background coding agent.
- submit() returns a task id immediately (async, never blocks the pipeline)
- state is persisted to disk so the MCP task server (separate process) can read/update it
- cancellation writes a cancel flag that the runner polls; cancelled tasks never speak stale results
"""
import asyncio
import json
import logging
import threading
from typing import Callable, Optional

from agent.task_events import Task, TaskState, new_task_id, dumps

log = logging.getLogger("coordinator")

# Events the voice agent should announce via TTS (not every low-level event)
ANNOUNCED_STATES = {TaskState.RUNNING, TaskState.AWAITING_APPROVAL, TaskState.COMPLETED, TaskState.FAILED}


class TaskStore:
    """Thread-safe JSON file store shared between the agent process and MCP servers."""

    def __init__(self, path):
        self.path = str(path)

    def _read(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)

    def upsert(self, task: Task) -> None:
        with threading.Lock():
            data = self._read()
            data[task.id] = task.to_dict()
            self._write(data)

    def get(self, task_id: str) -> Optional[Task]:
        d = self._read().get(task_id)
        return Task.from_dict(d) if d else None

    def list(self) -> list:
        return [Task.from_dict(d) for d in self._read().values()]

    def cancel_flag(self, task_id: str) -> str:
        return self.path + f".{task_id}.cancel"

    def is_cancel_requested(self, task_id: str) -> bool:
        import os
        return os.path.exists(self.cancel_flag(task_id))

    def request_cancel(self, task_id: str) -> bool:
        task = self.get(task_id)
        if not task:
            return False
        if task.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            return False
        with open(self.cancel_flag(task_id), "w") as f:
            f.write("cancel")
        task.state = TaskState.CANCELLED
        task.add_event("cancel_requested")
        self.upsert(task)
        return True

    def clear_cancel_flag(self, task_id: str) -> None:
        import os
        try:
            os.remove(self.cancel_flag(task_id))
        except FileNotFoundError:
            pass


class Coordinator:
    def __init__(self, store: TaskStore, runner_factory: Callable[[], "object"]):
        self.store = store
        self.runner_factory = runner_factory  # returns object with async run(task, on_event)
        self._tasks: dict = {}
        self._processes: dict = {}  # task_id -> runner handle
        self._announce: Optional[Callable[[str], None]] = None  # TTS sink

    def on_announce(self, sink: Callable[[str], None]) -> None:
        """Register the function used to speak announcements (TTS in voice mode, print in text mode)."""
        self._announce = sink

    def _announce_msg(self, text: str) -> None:
        if self._announce:
            try:
                self._announce(text)
            except Exception:
                log.exception("announce failed")

    async def submit(self, description: str) -> str:
        task = Task(id=new_task_id(), description=description)
        self._tasks[task.id] = task
        self.store.upsert(task)
        asyncio.create_task(self._run(task))
        return task.id

    async def status(self, task_id: str) -> Optional[Task]:
        return self.store.get(task_id)

    async def cancel(self, task_id: str) -> bool:
        ok = self.store.request_cancel(task_id)
        runner = self._processes.get(task_id)
        if runner and hasattr(runner, "abort"):
            try:
                await runner.abort()
            except Exception:
                log.exception("abort failed")
        if ok:
            self._announce_msg(f"Okay, cancelling task {task_id}.")
        return ok

    async def _run(self, task: Task) -> None:
        runner = self.runner_factory()
        self._processes[task.id] = runner

        def on_event(kind: str, detail: str = "") -> None:
            t = self.store.get(task.id)
            if t is None or t.state == TaskState.CANCELLED:
                return  # never propagate stale events after cancellation
            t.add_event(kind, detail)
            if kind == "progress":
                t.progress = detail
            self.store.upsert(t)
            if t.state != TaskState.CANCELLED and kind in ("started", "awaiting_approval", "completed", "failed"):
                self._announce_msg(self._humanize(t, kind, detail))

        try:
            t = self.store.get(task.id)
            t.state = TaskState.RUNNING
            t.add_event("started", "task picked up by background agent")
            self.store.upsert(t)

            result = await runner.run(task, on_event)

            t = self.store.get(task.id)
            if t.state == TaskState.CANCELLED:
                return
            t.result = str(result)[:2000]
            t.state = TaskState.COMPLETED
            t.add_event("completed", t.result)
            self.store.upsert(t)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            t = self.store.get(task.id)
            if t.state == TaskState.CANCELLED:
                return
            t.state = TaskState.FAILED
            t.add_event("failed", f"{type(e).__name__}: {e}")
            self.store.upsert(t)
            self._announce_msg(f"Heads up, task {task.id} failed. I can retry it if you want.")
        finally:
            self._processes.pop(task.id, None)
            self.store.clear_cancel_flag(task.id)

    @staticmethod
    def _humanize(task: Task, kind: str, detail: str) -> str:
        if kind == "started":
            return f"Okay, working on it: {task.description[:80]}"
        if kind == "awaiting_approval":
            return f"Task {task.id} needs your approval: {detail}"
        if kind == "completed":
            return f"Task {task.id} is done. {task.result[:120]}"
        if kind == "failed":
            return f"Task {task.id} failed. {detail[:120]}"
        return ""
