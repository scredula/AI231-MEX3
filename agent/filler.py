"""Contextual voice fillers while the slow model thinks / tools run.

A fast model (FILLER_MODEL, no thinking) generates a one-line spoken acknowledgment
that is pushed as a TTSSpeakFrame — spoken to the user but never added to the LLM
context, so the main model's response is unaffected. Falls back to canned phrases
if the filler model is slow or unavailable. One filler per turn, cancellable the
instant the real answer's first token or bot speech arrives.
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
}

FILLER_SYSTEM = (
    "You generate tiny spoken acknowledgments for a voice assistant while its main "
    "model prepares a full answer. Reply with exactly ONE short sentence, maximum 12 "
    "words, casual and conversational, acknowledging the user's message. No emojis, "
    "no lists, no quotes, no question marks at the end. Examples: 'Let me think that "
    "through.', 'Good question, one moment.', 'Checking on that now.'"
)

FILLER_TOOLS = set(TOOL_CANNED) | {"get_weather", "get_time"}


class FillerSpeaker:
    def __init__(self, speak: Callable[[str], "asyncio.Future"],
                 client: Optional[OpenRouterClient] = None,
                 thinking_secs: float = None, tool_secs: float = None,
                 enabled: bool = True):
        self.speak = speak  # async fn(text) — pushes TTSSpeakFrame into the pipeline
        self.enabled = enabled and Config.FILLER_ENABLED
        self.thinking_secs = thinking_secs if thinking_secs is not None else Config.THINKING_FILLER_SECS
        self.tool_secs = tool_secs if tool_secs is not None else Config.FILLER_AFTER_SECS
        self.client = client or OpenRouterClient(model=Config.FILLER_MODEL)
        self._pending: dict = {}        # turn_id -> (asyncio.Task, kind)
        self._spoken: set = set()       # turn_ids that already got a filler

    # ---- hooks called from the pipeline observer ----
    def on_llm_request(self, turn_id: int, user_text: str, already_spoken: bool = False) -> None:
        if not self.enabled or already_spoken:
            return
        # Don't re-arm while a tool filler is already pending/queued — the tool
        # phase is the reason we're slow; a thinking filler would reset the clock.
        if turn_id in self._pending:
            return
        self._start(turn_id, self._thinking(turn_id, user_text), "thinking")

    def on_tool_running(self, turn_id: int, tool_name: str) -> None:
        if not self.enabled or turn_id in self._spoken:
            return
        if tool_name not in FILLER_TOOLS:
            return
        # Re-arm: an earlier thinking filler may have been cancelled by first-token
        # text (the model thinking aloud between tool calls). Tool phases are long.
        self._start(turn_id, self._tool(turn_id, tool_name), "tool")

    def on_first_token(self, turn_id: int) -> None:
        # Only cancel thinking fillers: streamed text means the model is producing
        # the real answer. Tool fillers must survive — tokens between tool calls
        # don't reach the user while the tool is still running.
        self.cancel(turn_id, only="thinking")

    def on_bot_speech(self, turn_id: int) -> None:
        self.cancel(turn_id)

    def on_tool_done(self, turn_id: int) -> None:
        """Tool finished — cancel its pending filler (instant tools never speak one)."""
        self.cancel(turn_id, only="tool")

    def on_new_turn(self, turn_id: int) -> None:
        self.cancel(turn_id)

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
        if turn_id in self._spoken:
            return
        self._pending[turn_id] = (asyncio.get_event_loop().create_task(coro), kind)

    # ---- tasks ----
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
            await self._speak_filler(turn_id, f"(the user asked: {tool_name})", fallback)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("tool filler failed")

    async def _speak_filler(self, turn_id: int, user_text: str, fallback) -> None:
        if turn_id in self._spoken:
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
            if not text or len(text) > 140 or "\n" in text:
                text = None  # model misbehaved — use canned
        except (LLMError, asyncio.TimeoutError, Exception) as e:
            log.debug("filler model unavailable (%s) — using canned", e)
        if turn_id in self._spoken:
            return  # raced with a real response
        self._spoken.add(turn_id)
        if not text:
            text = random.choice(fallback) if isinstance(fallback, list) else fallback
        await self.speak(text)