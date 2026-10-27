"""Voice fillers while the slow model thinks / tools run — CANNED ONLY.

No filler model: fillers are short fixed phrases spoken via TTSSpeakFrame (never
added to the LLM context), so behavior is fully deterministic.

Two modes (FILLER_ALWAYS):
- always=1 (default): immediately after a transcript is recognized, a canned
  acknowledgment is spoken at once. The real answer plays after it (pipecat
  serializes TTS contexts). Mid-turn, "still working" fillers re-arm after each
  utterance while tools are pending.
- always=0: legacy threshold behavior — a canned filler after THINKING_FILLER_SECS.

Hard cap per turn: FILLER_MAX_PER_TURN. Kill switch: FILLER_ENABLED=0.
"""
import asyncio
import logging
import random
from typing import Callable, Optional

from agent.config import Config

log = logging.getLogger("filler")

CANNED_THINKING = [
    "One moment.", "Let me think about that.", "Just a second.",
    "Hmm, give me a second.", "Working on that one.",
]

TOOL_CANNED = {
    "search_web": "Let me search for that.",
    "get_weather": "Checking the weather now.",
    "run_command": "Running that for you.",
    "submit_task": "Kicking that off in the background.",
    "build_website": "Starting the build now.",
    "read_file": "Taking a look.",
    "edit_file": "Making that change.",
    "create_file": "Writing that file.",
    "_still_working": "Still working on that — one more moment.",
}

FILLER_TOOLS = set(TOOL_CANNED) | {"get_weather", "get_time"}


class FillerSpeaker:
    """Canned-only filler speaker. No LLM calls — fully deterministic."""

    def __init__(self, speak: Callable[[str], "asyncio.Future"],
                 thinking_secs: float = None, tool_secs: float = None,
                 delay_secs: float = None,
                 enabled: bool = True, always: Optional[bool] = None,
                 max_per_turn: int = None):
        self.speak = speak  # async fn(text) — pushes TTSSpeakFrame into the pipeline
        self.enabled = enabled and Config.FILLER_ENABLED
        self.always = Config.FILLER_ALWAYS if always is None else always
        self.thinking_secs = thinking_secs if thinking_secs is not None else Config.THINKING_FILLER_SECS
        self.tool_secs = tool_secs if tool_secs is not None else Config.FILLER_AFTER_SECS
        # always-mode: wait this long after the transcript; if the LLM hasn't
        # started responding by then, speak the canned filler
        self.delay_secs = delay_secs if delay_secs is not None else Config.FILLER_DELAY_SECS
        self.max_per_turn = max_per_turn if max_per_turn is not None else Config.FILLER_MAX_PER_TURN
        self._pending: dict = {}          # turn_id -> (asyncio.Task, kind)
        self._counts: dict = {}           # turn_id -> fillers spoken

    # ---- hooks called from the pipeline observer / LLM wrapper ----
    def on_llm_request(self, turn_id: int, user_text: str, already_spoken: bool = False) -> None:
        if not self.enabled or self.always:
            return  # always-mode arms the filler at the transcript instead
        if already_spoken or turn_id in self._pending:
            return
        self._start(turn_id, self._thinking(turn_id), "thinking")

    def on_transcript(self, turn_id: int, text: str = "") -> None:
        """Real (recognized) speech arrived. In always-mode this triggers the
        immediate canned filler; in threshold mode it does nothing."""
        for tid in list(self._pending):
            if tid != turn_id:
                self.cancel(tid)
        if self.always and self.enabled and not self._at_cap(turn_id):
            self._start(turn_id, self._immediate(turn_id), "immediate")

    def on_first_token(self, turn_id: int) -> None:
        """The LLM is responding — no filler needed, cancel everything pending
        (thinking, tool, and the delayed immediate filler)."""
        self.cancel(turn_id)

    def on_tool_running(self, turn_id: int, tool_name: str) -> None:
        if not self.enabled or tool_name not in FILLER_TOOLS:
            return
        if turn_id in self._pending:
            return  # something is already armed for this turn
        if self.always and self._counts.get(turn_id, 0) > 0:
            return  # always-mode: a filler already spoke this turn — no mid-turn
            # tool fillers (they delay short tools and add chatter)
        self._start(turn_id, self._canned_only(turn_id, TOOL_CANNED.get(tool_name, "One moment.")), "tool")

    def on_bot_speech(self, turn_id: int) -> None:
        """The bot started speaking real audio — cancel any armed fillers."""
        self.cancel(turn_id)

    def on_tool_done(self, turn_id: int) -> None:
        """Tool finished — cancel its pending filler (instant tools never speak one)."""
        self.cancel(turn_id, only="tool")

    def on_bot_speech_end(self, turn_id: int, work_pending: bool) -> None:
        """The bot finished an utterance but work may still be in flight — re-arm
        the filler clock so the user never waits long in silence mid-turn."""
        if not self.enabled or not work_pending or self._at_cap(turn_id):
            return
        if turn_id in self._pending:
            return
        self._start(turn_id, self._canned_only(turn_id, TOOL_CANNED["_still_working"]), "tool")

    # ---- internals ----
    def _at_cap(self, turn_id: int) -> bool:
        return self._counts.get(turn_id, 0) >= self.max_per_turn

    def cancel(self, turn_id: int, only: Optional[str] = None) -> None:
        entry = self._pending.pop(turn_id, None)
        if entry:
            task, kind = entry
            if only is None or kind == only:
                task.cancel()
            else:
                self._pending[turn_id] = entry  # keep the other kind pending

    def _start(self, turn_id: int, coro, kind: str) -> None:
        self.cancel(turn_id)
        self._pending[turn_id] = (asyncio.get_event_loop().create_task(coro), kind)

    def _speak_allowed(self, turn_id: int) -> bool:
        if self._at_cap(turn_id):
            return False
        entry = self._pending.get(turn_id)
        # a pending entry belonging to THIS running task is fine; another armed
        # filler for the same turn means something newer took over — stay silent
        if entry and entry[0] is not asyncio.current_task():
            return False
        return True

    async def _speak_canned(self, turn_id: int, phrase: str) -> None:
        if not self._speak_allowed(turn_id):
            return
        self._pending.pop(turn_id, None)
        self._counts[turn_id] = self._counts.get(turn_id, 0) + 1
        await self.speak(phrase)

    # ---- tasks ----
    async def _immediate(self, turn_id: int) -> None:
        try:
            # wait for the LLM to respond; if it does (first token / bot speech),
            # this task is cancelled before it ever speaks
            await asyncio.sleep(self.delay_secs)
            await self._speak_canned(turn_id, random.choice(CANNED_THINKING))
        except asyncio.CancelledError:
            pass

    async def _thinking(self, turn_id: int) -> None:
        try:
            await asyncio.sleep(self.thinking_secs)
            await self._speak_canned(turn_id, random.choice(CANNED_THINKING))
        except asyncio.CancelledError:
            pass

    async def _canned_only(self, turn_id: int, phrase: str) -> None:
        try:
            await asyncio.sleep(self.tool_secs)
            await self._speak_canned(turn_id, phrase)
        except asyncio.CancelledError:
            pass