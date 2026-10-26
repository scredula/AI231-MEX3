"""Pi coding agent (pi.dev / badlogic pi-mono) adapter.

Talks to `pi --mode rpc --no-session` as a subprocess using strict JSONL framing:
- send {"id": ..., "type": "prompt", "message": ...}
- read session events from stdout (agent_start / message_update / tool_execution_* / agent_end...)
- cancellation via the "abort" command
Model/provider configured via CLI flags from env (CODING_MODEL). OpenRouter is a built-in
pi provider that reads $OPENROUTER_API_KEY.

If pi is not installed, use PythonFallbackAdapter instead (see runner.py).
"""
import asyncio
import json
import logging
import shutil
from typing import AsyncIterator, Optional

from agent.config import Config

log = logging.getLogger("pi_adapter")


class PiAdapter:
    name = "pi"

    def __init__(self, workspace, model: str = None):
        self.workspace = str(workspace)
        self.model = model or Config.CODING_MODEL
        self.proc: Optional[asyncio.subprocess.Process] = None
        self._req_counter = 0
        self._pending: dict = {}  # request id -> Future
        self._reader_task: Optional[asyncio.Task] = None
        self._aborted = False
        self._settled_hook = None

    def available(self) -> bool:
        return self._resolve_executable() is not None

    def _resolve_executable(self):
        """Locate the pi executable. On Windows npm installs produce pi.cmd shims,
        which must be launched through cmd.exe."""
        exe = shutil.which("pi.cmd") or shutil.which("pi.bat") or shutil.which("pi")
        if not exe:
            return None
        if exe.lower().endswith((".cmd", ".bat")):
            return ["cmd", "/c", exe]
        return [exe]

    async def start(self) -> None:
        cmd = self._resolve_executable()
        if not cmd:
            raise RuntimeError("pi CLI not found")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd, "--mode", "rpc", "--no-session",
                "--provider", "openrouter", "--model", self.model,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(f"cannot spawn pi: {e}") from e
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        buffer = b""
        while True:
            chunk = await self.proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                self._handle(rec)

    def _handle(self, rec: dict) -> None:
        if rec.get("type") == "response" and rec.get("id") in self._pending:
            fut = self._pending.pop(rec["id"])
            if not fut.done():
                fut.set_result(rec)
        else:
            self._last_event = rec
            if rec.get("type") == "agent_settled" and self._settled_hook:
                self._settled_hook(rec)
            self._emit(rec)

    # event plumbing -------------------------------------------------------
    def _emit(self, rec: dict) -> None:
        if self._on_event and rec.get("type") not in ("response",):
            try:
                self._on_event(rec.get("type", "event"), json.dumps(rec)[:200])
            except Exception:
                log.exception("event sink failed")

    _on_event = None

    def set_event_sink(self, sink) -> None:
        self._on_event = sink

    async def _rpc(self, command: dict, timeout: float = 30) -> dict:
        self._req_counter += 1
        rid = f"req-{self._req_counter}"
        command = {"id": rid, **command}
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        self.proc.stdin.write((json.dumps(command) + "\n").encode())
        await self.proc.stdin.drain()
        return await asyncio.wait_for(fut, timeout)

    # high-level API -------------------------------------------------------
    async def run_task(self, prompt: str, timeout: float = 900) -> str:
        """Send a task prompt and collect the agent's final assistant text."""
        resp = await self._rpc({"type": "prompt", "message": prompt}, timeout=30)
        if not resp.get("success"):
            raise RuntimeError(f"pi rejected prompt: {resp}")
        settled = asyncio.Event()

        def on_settled(_rec: dict) -> None:
            settled.set()

        self._settled_hook = on_settled
        try:
            await asyncio.wait_for(settled.wait(), timeout)
        except asyncio.TimeoutError:
            log.warning("pi run timed out after %ss — aborting", timeout)
            await self.abort()
        text = ""
        try:
            data = (await self._rpc({"type": "get_last_assistant_text"}, timeout=30)).get("data") or {}
            text = data.get("text") or ""
        except (asyncio.TimeoutError, Exception) as e:
            log.warning("get_last_assistant_text failed: %s", e)
        return text

    async def abort(self) -> None:
        self._aborted = True
        try:
            await self._rpc({"type": "abort"}, timeout=15)
        except Exception:
            pass

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
        if self._reader_task:
            self._reader_task.cancel()
