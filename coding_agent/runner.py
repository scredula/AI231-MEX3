"""Background task runner.

Two interchangeable adapters:
- PiAdapter: pi.dev coding agent over RPC (preferred; strong multi-step coding agent)
- PythonFallbackAdapter: local OpenRouter tool-calling loop with restricted tools

Both run inside the coordinator's asyncio task and honour cancellation flags.
"""
import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path

from agent.config import Config
from agent.task_events import Task
from coding_agent.adapter import PiAdapter
from coding_agent.policies import ExecutionPolicy, PolicyViolation, DEFAULT_ALLOWLIST, MAX_RETRIES
from services.llm_client import OpenRouterClient, LLMError

log = logging.getLogger("runner")

TASK_SYSTEM_PROMPT = (
    "You are a background coding agent working inside a restricted workspace. "
    "Complete the user's task step by step: create files, run checks, fix errors and retry "
    "(at most %d retries). Never access paths outside the workspace. "
    "Never print or embed credentials. When finished, reply with a short summary of what you built."
    % MAX_RETRIES
)


def build_prompt(task: Task) -> str:
    return (
        f"Workspace task: {task.description}\n"
        "Work inside the current directory only. Create/verify the required files, "
        "run any local checks (tests, builds), fix errors yourself, then report the result briefly."
    )


class _BaseRunner:
    def __init__(self, workspace):
        self.policy = ExecutionPolicy(workspace)

    def _cancelled(self, task: Task) -> bool:
        from agent.config import Config
        store_task = task
        # cheap check against cancel flag on disk
        flag = str(Config.TASKS_DB_PATH) + f".{task.id}.cancel"
        import os
        return os.path.exists(flag)


class PiRunner(_BaseRunner):
    """Runs tasks through the Pi coding agent subprocess."""

    name = "pi"

    def __init__(self, workspace):
        super().__init__(workspace)
        self.adapter = PiAdapter(workspace)
        if not self.adapter.available():
            raise RuntimeError("pi CLI not found — install @mariozechner/pi-coding-agent")

    async def run(self, task: Task, on_event) -> str:
        self.adapter.set_event_sink(lambda k, d: on_event("progress", d) if not self._cancelled(task) else None)
        try:
            await self.adapter.start()
        except RuntimeError as e:
            raise RuntimeError(f"pi runner unavailable: {e}") from e
        try:
            return await self.adapter.run_task(build_prompt(task))
        finally:
            await self.adapter.stop()
            if self._cancelled(task):
                on_event("cancelled", "task aborted by user")

    async def abort(self) -> None:
        await self.adapter.abort()


class PythonFallbackRunner(_BaseRunner):
    """OpenRouter tool-calling loop with a restricted local toolset. Works on any Python host."""

    name = "fallback"

    TOOLS = [
        {"type": "function", "function": {"name": "write_file", "description": "Write a text file inside the workspace",
         "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "run_command", "description": "Run an allowlisted shell command in the workspace",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    ]

    def __init__(self, workspace):
        super().__init__(workspace)
        self.client = OpenRouterClient()

    async def run(self, task: Task, on_event) -> str:
        messages = [
            {"role": "system", "content": TASK_SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(task)},
        ]
        for step in range(MAX_RETRIES * 4):
            if self._cancelled(task):
                raise asyncio.CancelledError()
            try:
                msg = await self.client.chat(messages, tools=self.TOOLS)
            except LLMError as e:
                if step >= MAX_RETRIES - 1:
                    raise
                on_event("progress", f"LLM error, retrying: {e}")
                await asyncio.sleep(2 * (step + 1))
                continue
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content", "")
            for call in calls:
                result = await self._exec_tool(call, on_event)
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": result if isinstance(result, str) else json.dumps(result)})
        raise RuntimeError("task exceeded step limit")

    async def _exec_tool(self, call: dict, on_event) -> str:
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            return "error: bad arguments"
        try:
            if name == "write_file":
                path = self.policy.check_path(args["path"])
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(args["content"], encoding="utf-8")
                on_event("progress", f"wrote {path.name}")
                return f"ok: wrote {path}"
            if name == "run_command":
                cmd = args["command"]
                self.policy.check_command(cmd)
                proc = subprocess.run(cmd, shell=True, cwd=str(self.policy.workspace),
                                      capture_output=True, text=True, timeout=self.policy.timeout)
                out = (proc.stdout + proc.stderr)[-2000:]
                on_event("progress", f"ran: {cmd[:60]}")
                return f"exit={proc.returncode}\n{out}"
            return f"error: unknown tool {name}"
        except PolicyViolation as e:
            return f"error: policy violation — {e}"
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {self.policy.timeout}s"


def make_runner_factory(workspace):
    """Prefer Pi if installed; fall back to the Python tool loop."""
    def factory():
        try:
            return PiRunner(workspace)
        except RuntimeError:
            log.info("pi not available, using Python fallback runner")
            return PythonFallbackRunner(workspace)
    return factory
