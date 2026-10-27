"""Voice fillers while the slow model thinks / tools run.

Two modes (FILLER_ALWAYS):
- always=1 (default): immediately after a transcript is recognized, the fast model
  (FILLER_MODEL, no thinking) generates a short acknowledgment that is spoken at once.
  The real answer plays after it (pipecat serializes TTS contexts). Mid-turn, canned
  "still working" fillers re-arm after each utterance while work is pending.
- always=0: legacy threshold behavior — thinking fillers after THINKING_FILLER_SECS.

Fillers are TTSSpeakFrame-only: spoken to the user, never added to the LLM context.
Canned fallback if the filler model is slow/unavailable. Hard cap: FILLER_MAX_PER_TURN.
"""
import asyncio
import logging
import random
from typing import Callable, Optional

from agent.config import Config
from services.llm_client import OpenRouterClient, LLMError

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

FILLER_SYSTEM = (
    "You generate tiny spoken acknowledgments for a voice assistant while its main "
    "model prepares a full answer. Reply with exactly ONE very short sentence, "
    "maximum 8 words, casual and conversational, acknowledging what the user is "
    "waiting for. No emojis, no lists, no quotes, no question marks. Never apologize, "
    "never react to anything except the user's request itself, and never mention "
    "being an AI or another model. Examples: 'Let me look that up.', 'Good question, "
    "one moment.', 'Checking on that now.'"
)


class FillerSpeaker:
    def __init__(self, speak: Callable[[str], "asyncio.Future"],
                 client: Optional[OpenRouterClient] = None,
                 thinking_secs: float = None, tool_secs: float = None,
                 enabled: bool = True, always: Optional[bool] = None,
                 max_per_turn: int = None):
        self.speak = speak  # async fn(text) — pushes TTSSpeakFrame into the pipeline
        self.enabled = enabled and Config.FILLER_ENABLED
        self.always = Config.FILLER_ALWAYS if always is None else always
        self.thinking_secs = thinking_secs if thinking_secs is not None else Config.THINKING_FILLER_SECS
        self.tool_secs = tool_secs if tool_secs is not None else Config.FILLER_AFTER_SECS
        self.max_per_turn = max_per_turn if max_per_turn is not None else Config.FILLER_MAX_PER_TURN
        self.client = client or OpenRouterClient(model=Config.FILLER_MODEL)
        self._pending: dict = {}          # turn_id -> (asyncio.Task, kind)
        self._counts: dict = {}           # turn_id -> fillers spoken
        self._immediate_spoken: set = set()
        self._last_transcript: str = ""   # latest real user text

    # ---- hooks called from the pipeline observer / LLM wrapper ----
    def on_llm_request(self, turn_id: int, user_text: str, already_spoken: bool = False) -> None:
        if not self.enabled or self.always:
            return  # always-mode arms the filler at the transcript instead
        if already_spoken or turn_id in self._pending:
            return
        self._start(turn_id, self._thinking(turn_id, user_text), "thinking")

    def on_transcript(self, turn_id: int, text: str = "") -> None:
        """Real (recognized) speech arrived. In always-mode this triggers the
        immediate filler; in threshold mode it arms the thinking filler."""
        if text:
            self._last_transcript = text
        for tid in list(self._pending):
            if tid != turn_id:
                self.cancel(tid)
        if self.always and self.enabled and not self._at_cap(turn_id):
            self._start(turn_id, self._immediate(turn_id), "immediate")

    def on_first_token(self, turn_id: int) -> None:
        """Real answer audio is coming. Cancel thinking fillers (threshold mode)
        and tool fillers queued behind the response. The immediate filler
        (always-mode) always speaks."""
        self.cancel(turn_id, only="thinking")
        self.cancel(turn_id, only="tool")

    def on_tool_running(self, turn_id: int, tool_name: str) -> None:
        if not self.enabled or tool_name not in FILLER_TOOLS:
            return
        if turn_id in self._pending:
            return  # something is already armed for this turn
        if turn_id in self._counts and self._counts[turn_id] > 0:
            # immediate filler already covered the start — canned only
            self._start(turn_id, self._canned_only(turn_id, TOOL_CANNED.get(tool_name, "One moment.")), "tool")
        else:
            self._start(turn_id, self._tool(turn_id, tool_name), "tool")

    def on_bot_speech(self, turn_id: int) -> None:
        self.cancel(turn_id, only="tool")

    def on_tool_done(self, turn_id: int) -> None:
        """Tool finished — cancel its pending filler (instant tools never speak one)."""
        self.cancel(turn_id, only="tool")

    def on_bot_speech_end(self, turn_id: int, work_pending: bool) -> None:
        """The bot finished an utterance but work may still be in flight — re-arm
        the filler clock so the user never waits long in silence mid-turn.
        Canned only: the filler model must never 'react' to fake input."""
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

    def _finish_speaking(self, turn_id: int) -> None:
        entry = self._pending.get(turn_id)
        if entry and entry[0] is asyncio.current_task():
            self._pending.pop(turn_id, None)

    async def _speak_canned(self, turn_id: int, phrase: str) -> None:
        if not self._speak_allowed(turn_id):
            return
        self._finish_speaking(turn_id)
        self._counts[turn_id] = self._counts.get(turn_id, 0) + 1
        await self.speak(phrase)

    async def _speak_filler(self, turn_id: int, user_text: str, fallback) -> None:
        if not self._speak_allowed(turn_id):
            return
        text = None
        try:
            msg = await asyncio.wait_for(
                self.client.chat(
                    [{"role": "system", "content": FILLER_SYSTEM},
                     {"role": "user", "content": user_text[:300]}],
                    temperature=0.9),
                timeout=4.0)
            text = (msg.get("content") or "").strip().strip('"')
            if not text or len(text) > 100 or "\n" in text:
                text = None  # model misbehaved — use canned
        except (LLMError, asyncio.TimeoutError, Exception) as e:
            log.debug("filler model unavailable (%s) — using canned", e)
        if not self._speak_allowed(turn_id):
            return  # raced with a real response / cap
        self._finish_speaking(turn_id)
        self._counts[turn_id] = self._counts.get(turn_id, 0) + 1
        if not text:
            text = random.choice(fallback) if isinstance(fallback, list) else fallback
        await self.speak(text)

    # ---- tasks ----
    async def _immediate(self, turn_id: int) -> None:
        try:
            user_text = self._last_transcript
            if user_text:
                await self._speak_filler(turn_id, user_text, CANNED_THINKING)
            else:
                await self._speak_canned(turn_id, random.choice(CANNED_THINKING))
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("immediate filler failed")

    async def _thinking(self, turn_id: int, user_text: str) -> None:
        try:
            await asyncio.sleep(self.thinking_secs)
            await self._speak_filler(turn_id, user_text, CANNED_THINKING)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("thinking filler failed")

    async def _tool(self, turn_id: int, tool_name: str) -> None:
        try:
            await asyncio.sleep(self.tool_secs)
            fallback = TOOL_CANNED.get(tool_name, "One moment.")
            user_text = self._last_transcript or ""
            if user_text:
                await self._speak_filler(turn_id, user_text, fallback)
            else:
                await self._speak_canned(turn_id, fallback)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("tool filler failed")

    async def _canned_only(self, turn_id: int, phrase: str) -> None:
        try:
            await asyncio.sleep(self.tool_secs)
            await self._speak_canned(turn_id, phrase)
        except asyncio.CancelledError:
            pass