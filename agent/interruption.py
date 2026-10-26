"""Interruption helpers.

Pipecat handles barge-in natively: when the user speaks while the bot is talking, the
framework emits InterruptionFrame / StartInterruptionFrame upstream, cancels TTS playback
and marks the assistant message as interrupted. This module records the timestamps used by
the latency benchmark:

- `InterruptionTracker` observes BotStartedSpeakingFrame / UserStartedSpeakingFrame and
  exposes last_speech_onset(), last_bot_speech_start(), interruption latency
  (bot speech start → user interrupt) for benchmark/run_benchmark.py.
"""
import time
from dataclasses import dataclass, field


@dataclass
class InterruptionEvent:
    t_bot_speech_start: float
    t_user_interrupt: float

    @property
    def latency(self) -> float:
        return self.t_user_interrupt - self.t_bot_speech_start


@dataclass
class InterruptionTracker:
    events: list = field(default_factory=list)
    _bot_speech_start: float = 0.0
    _user_speaking: bool = False
    last_speech_onset: float = 0.0

    def on_user_started_speaking(self) -> None:
        now = time.monotonic()
        self.last_speech_onset = now
        self._user_speaking = True
        if self._bot_speech_start:
            self.events.append(InterruptionEvent(self._bot_speech_start, now))
            self._bot_speech_start = 0.0

    def on_user_stopped_speaking(self) -> None:
        self._user_speaking = False

    def on_bot_started_speaking(self) -> None:
        self._bot_speech_start = time.monotonic()
