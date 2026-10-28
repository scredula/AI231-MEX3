"""Timed reminders with JSON persistence and a polling scheduler.

The store is file-backed (data/reminders.json) so BOTH the voice agent (which
speaks reminders through TTS) and the MCP tools server can add/list/cancel —
the agent's poll loop picks up reminders added by any process.

Reminder dict: {id, text, due (unix seconds), created, fired}
"""
import asyncio
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("reminders")


class ReminderStore:
    def __init__(self, path: Path, on_fire=None, poll_secs: float = 5.0):
        self.path = Path(path)
        self.on_fire = on_fire          # async fn(reminder_dict) -> None
        self.poll_secs = poll_secs
        self._task = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    # ---- persistence ----
    def _read(self) -> list:
        try:
            return json.loads(self.path.read_text(encoding="utf-8") or "[]")
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, items: list) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # ---- api (sync, safe from any process via the file) ----
    def add(self, text: str, in_minutes: float = 1.0) -> dict:
        text = (text or "").strip()[:200]
        if not text:
            raise ValueError("reminder text is required")
        if not 0.02 <= in_minutes <= 60 * 24 * 7:
            raise ValueError("in_minutes must be between ~1 second and 7 days")
        items = self._read()
        rem = {"id": int(time.time() * 1000) % 1_000_000,
               "text": text,
               "due": round(time.time() + in_minutes * 60, 1),
               "created": round(time.time(), 1),
               "fired": False}
        items.append(rem)
        self._write(items)
        return rem

    def list_active(self) -> list:
        now = time.time()
        return sorted((r for r in self._read() if not r.get("fired") and r["due"] > now),
                      key=lambda r: r["due"])

    def cancel(self, reminder_id: int) -> dict:
        items = self._read()
        for r in items:
            if r["id"] == reminder_id and not r.get("fired"):
                r["fired"] = True          # mark cancelled (fired=True, flagged)
                r["cancelled"] = True
                self._write(items)
                return r
        raise ValueError(f"no active reminder with id {reminder_id}")

    # ---- scheduler (runs in the agent process) ----
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_event_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                now = time.time()
                items = self._read()
                changed = False
                for r in items:
                    if not r.get("fired") and not r.get("cancelled") and r["due"] <= now:
                        r["fired"] = True
                        changed = True
                        log.info("reminder fired: %s", r["text"])
                        if self.on_fire:
                            try:
                                await self.on_fire(r)
                            except Exception as e:  # never kill the loop
                                log.error("reminder on_fire failed: %s", e)
                if changed:
                    self._write(items)
            except Exception as e:
                log.error("reminder loop error: %s", e)
            await asyncio.sleep(self.poll_secs)
