"""Observability: captures every stage of the conversation and feeds the dashboard.

- DashboardObserver: pipecat frame observer — user speech, transcripts, tool calls,
  bot speech, interruptions; stamps monotonic latencies per turn.
- ObservableLLMService: OpenAILLMService subclass that publishes the exact messages
  passed into the LLM on every request.
- EventBus: thread-safe pub/sub; the dashboard (separate thread) streams events via SSE.
"""
import json
import logging
import queue
import threading
import time
from typing import Optional

from agent.config import Config

from pipecat.frames.frames import (
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame, ErrorFrame,
    FunctionCallInProgressFrame, FunctionCallResultFrame, FunctionCallsStartedFrame,
    InterruptionFrame, LLMTextFrame, TranscriptionFrame, TTSSpeakFrame,
    UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
)

log = logging.getLogger("observability")


def _serialize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(str(c.get("text") or c.get("content") or c))
            else:
                parts.append(_serialize_content(c))
        return " | ".join(parts)
    return str(content)


class EventBus:
    """Thread-safe publish/subscribe with SSE-friendly string payloads."""

    def __init__(self, maxlen: int = 500):
        self._subs: list = []
        self._history: list = []
        self._history_max = maxlen
        self._lock = threading.Lock()

    def subscribe(self) -> queue.SimpleQueue:
        q = queue.SimpleQueue()
        with self._lock:
            for ev in self._history:
                q.put(ev)
            self._subs.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        event.setdefault("ts", round(time.time(), 3))
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max:]
            subs = list(self._subs)
        for q in subs:
            q.put(event)


class TurnTracker:
    """Tracks the current conversation turn and its latency milestones."""

    def __init__(self, on_event):
        self._on_event = on_event
        self.turn_id = 0
        self.t_user_speech = None
        self.t_speech_end = None
        self.t_transcript = None
        self.transcript_text = ""
        self.t_llm_request = None
        self.t_llm_first_token = None
        self.llm_text = ""
        self.t_bot_speech = None
        self.tts_contexts_done = 0
        self.interrupted = False

    def _lat(self, a, b) -> Optional[float]:
        return None if a is None or b is None else round(b - a, 3)

    def latencies(self) -> dict:
        return {
            "speech_to_transcript": self._lat(self.t_user_speech, self.t_transcript),
            "transcript_to_llm_request": self._lat(self.t_transcript, self.t_llm_request),
            "llm_request_to_first_token": self._lat(self.t_llm_request, self.t_llm_first_token),
            "llm_request_to_bot_speech": self._lat(self.t_llm_request, self.t_bot_speech),
            "speech_end_to_bot_speech": self._lat(self.t_speech_end, self.t_bot_speech),
            "speech_onset_to_bot_speech": self._lat(self.t_user_speech, self.t_bot_speech),
            "interrupted": self.interrupted,
        }

    def new_turn(self) -> None:
        self.turn_id += 1
        self.t_user_speech = time.monotonic()
        self.t_speech_end = self.t_transcript = self.t_llm_request = None
        self.t_llm_first_token = self.t_bot_speech = None
        self.transcript_text = self.llm_text = ""
        self.tts_contexts_done = 0
        self.interrupted = False
        self._on_event({"type": "turn_start", "turn": self.turn_id})

    def end_turn(self) -> None:
        self._on_event({"type": "turn_end", "turn": self.turn_id,
                        "user_text": self.transcript_text,
                        "assistant_text": self.llm_text,
                        "latencies": self.latencies()})


class DashboardObserver:
    """Pipecat observer: publish conversation events + latencies to the EventBus.

    Subclasses pipecat's BaseObserver (lifecycle + first-push-only delivery).
    """

    def __init__(self, bus: EventBus):
        from pipecat.observers.base_observer import BaseObserver

        class _PipecatBridge(BaseObserver):
            def __init__(inner_self, obs):
                super().__init__(observe_every_push=False)
                inner_self._obs = obs

            async def on_push_frame(inner_self, data) -> None:
                await inner_self._obs.on_push_frame(data)

        self.bus = bus
        self.tracker = TurnTracker(self._emit)
        self._tool_starts = {}  # tool_call_id -> monotonic start
        self.filler = None      # optional FillerSpeaker, set by the pipeline
        self._pending_filler_announcement = False
        self._bridge = _PipecatBridge(self)

    def pipecat_observer(self):
        """The actual BaseObserver instance to register with the pipeline worker."""
        return self._bridge

    def _emit(self, event: dict) -> None:
        self.bus.publish(event)

    async def on_push_frame(self, data) -> None:
        try:
            await self._handle(data)
        except Exception:
            log.exception("dashboard observer error")

    async def _handle(self, data) -> None:
        frame = data.frame
        now = time.monotonic()
        t = self.tracker
        tms = int(time.time() * 1000)
        elapsed = None if t.t_user_speech is None else int((now - t.t_user_speech) * 1000)
        if isinstance(frame, UserStartedSpeakingFrame):
            t.new_turn()
            self._emit({"type": "speech_start", "turn": t.turn_id, "t_ms": tms, "elapsed_ms": 0})
        elif isinstance(frame, UserStoppedSpeakingFrame):
            t.t_speech_end = now
            self._emit({"type": "speech_end", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed,
                        "speech_ms": int((now - t.t_user_speech) * 1000) if t.t_user_speech else None})
        elif isinstance(frame, TranscriptionFrame):
            t.t_transcript = now
            t.transcript_text = str(getattr(frame, "text", ""))
            if self.filler:
                self.filler.on_transcript(t.turn_id, t.transcript_text)  # real speech cancels stale fillers
            self._emit({"type": "transcript", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed, "text": t.transcript_text})
        elif isinstance(frame, InterruptionFrame):
            if t.t_bot_speech:
                t.interrupted = True
                self._emit({"type": "interruption", "turn": t.turn_id, "t_ms": tms,
                            "elapsed_ms": elapsed,
                            "interrupt_latency": self.tracker.latencies()["llm_request_to_bot_speech"]})
        elif isinstance(frame, FunctionCallsStartedFrame):
            calls = [{"name": getattr(c, "function_name", "?"),
                      "arguments": str(getattr(c, "arguments", ""))} for c in getattr(frame, "function_calls", [])]
            self._emit({"type": "tool_calls_started", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed, "calls": calls})
        elif isinstance(frame, FunctionCallInProgressFrame):
            call_id = str(getattr(frame, "tool_call_id", ""))
            self._tool_starts[call_id] = (now, tms)
            if self.filler:
                self.filler.on_tool_running(t.turn_id, getattr(frame, "function_name", "?"))
            self._emit({"type": "tool_running", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed,
                        "id": call_id,
                        "name": getattr(frame, "function_name", "?"),
                        "start_ms": tms,
                        "arguments": _safe_json(getattr(frame, "arguments", None))})
        elif isinstance(frame, FunctionCallResultFrame):
            result = getattr(frame, "result", "")
            call_id = str(getattr(frame, "tool_call_id", ""))
            start = self._tool_starts.pop(call_id, None)
            if self.filler:
                self.filler.on_tool_done(t.turn_id)  # instant tools don't need fillers
            self._emit({"type": "tool_result", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed,
                        "id": call_id,
                        "name": getattr(frame, "function_name", "?"),
                        "start_ms": start[1] if start else None,
                        "ok": "error" not in str(result)[:80].lower(),
                        "duration": round(now - start[0], 3) if start else None,
                        "result": str(result)[:2000]})
        elif isinstance(frame, LLMTextFrame):
            if self.filler:
                self.filler.on_first_token(t.turn_id)
            text = str(getattr(frame, "text", ""))
            if not t.t_llm_first_token:
                t.t_llm_first_token = now
                self._emit({"type": "first_token", "turn": t.turn_id, "t_ms": tms,
                            "elapsed_ms": elapsed, "model": Config.FAST_MODEL})
            t.llm_text += text
            self._emit({"type": "llm_delta", "turn": t.turn_id, "delta": text})
        elif isinstance(frame, BotStartedSpeakingFrame):
            if self.filler:
                self.filler.on_bot_speech(t.turn_id)
            t.t_bot_speech = now
            kind = "filler" if self._pending_filler_announcement else "response"
            self._pending_filler_announcement = False
            self._emit({"type": "bot_speech_start", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed, "kind": kind, "model": Config.FAST_MODEL,
                        "latencies": t.latencies()})
        elif isinstance(frame, BotStoppedSpeakingFrame):
            t.end_turn()
            # Work may still be in flight (tools/next LLM call) — re-arm the filler
            # clock so the user never waits long in silence mid-turn.
            if self.filler:
                self.filler.on_bot_speech_end(t.turn_id, work_pending=bool(self._tool_starts))
        elif isinstance(frame, TTSSpeakFrame):
            self._pending_filler_announcement = True
            self._emit({"type": "announcement", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed, "text": str(frame.text)[:300],
                        "model": Config.FILLER_MODEL})
        elif isinstance(frame, ErrorFrame):
            self._emit({"type": "error", "turn": t.turn_id, "t_ms": tms,
                        "elapsed_ms": elapsed, "error": str(frame.error)[:300]})


def _safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)[:400]
    except (TypeError, ValueError):
        return str(obj)[:400]


class ObservableLLMService:
    """Mixin that publishes the exact messages sent to the LLM on each request."""

    def _observability_hook(self, context):
        try:
            obs = getattr(self, "_observer", None)
            if obs is None:
                return
            msgs = []
            for m in context.get_messages():
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "?")
                content = getattr(m, "content", None)
                if content is None and isinstance(m, dict):
                    content = m.get("content")
                msgs.append({"role": str(role), "content": _serialize_content(content)[:1200]})
            obs.bus.publish({"type": "llm_request", "turn": obs.tracker.turn_id,
                             "model": getattr(self, "model", None) and str(getattr(self, "model", "")),
                             "messages": msgs})
            obs.tracker.t_llm_request = time.monotonic()
        except Exception:
            log.exception("observability llm hook failed")