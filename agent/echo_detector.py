"""Echo detection: classify mic bursts as either the room echo of the bot's own
TTS playback or genuine user speech — by correlating the mic audio against the
audio we just played.

Why: the echo gate (mute mic during TTS) kills the "agent hears itself" bug but
also kills barge-in. This detector keeps the mic OPEN and decides per speech
burst: if the mic signal is a delayed copy of recently-played TTS audio (FFT
cross-correlation over a lag sweep), the whole burst is suppressed — no
interruption, no transcript, nothing enters the LLM context. If it doesn't
match the played audio, it's a real user speaking and flows through normally,
including mid-TTS interruptions.

A transcript backstop catches what the correlation misses: any user transcript
arriving shortly after bot speech that fuzzy-matches the bot's own recent text
is discarded.

Wiring (pipeline):
  transport.input() -> stt -> vad -> EchoDetectorProcessor -> user aggregator
  aggregators.assistant() -> TTSOutputTap -> transport.output()
The observer feeds bot text (LLMTextFrame deltas, TTSSpeakFrame) via on_bot_text.
"""
import asyncio
import difflib
import logging
import re
import time

import numpy as np
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor

log = logging.getLogger("echo")

_PUNCT = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    return _PUNCT.sub(" ", text.lower()).strip()


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Fast linear-interpolation resample (quality is fine for correlation)."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32)
    n_out = int(round(samples.size * dst_rate / src_rate))
    if n_out == 0:
        return np.zeros(0, np.float32)
    idx = np.linspace(0, samples.size - 1, n_out)
    return np.interp(idx, np.arange(samples.size), samples).astype(np.float32)


def _corr_score(seg: np.ndarray, ref: np.ndarray, max_lag: int | None = None) -> float:
    """Best normalized cross-correlation of `seg` against `ref` over ALL lags.

    The mic window can sit anywhere inside the reference ring buffer (its offset
    absorbs playback-buffer delay), so we sweep every lag up to
    ref.size - seg.size. Both float32 mono; returns the max Pearson-style
    coefficient: ~1 for a delayed copy (echo), ~0 for unrelated audio.
    """
    if seg.size < 400 or ref.size < seg.size:
        return 0.0
    seg = seg - seg.mean()
    seg_norm = np.sqrt((seg * seg).sum())
    if seg_norm < 1e-6:
        return 0.0
    nfft = 1 << int(np.ceil(np.log2(ref.size + seg.size)))
    fs = np.fft.rfft(seg, nfft)
    fr = np.fft.rfft(ref, nfft)
    xcorr = np.fft.irfft(fr * np.conj(fs), nfft)  # x[k] = sum ref[n+k]*seg[n] — peak at the mic-window offset within ref
    lags = ref.size - seg.size if max_lag is None else min(max_lag, ref.size - seg.size)
    if lags <= 0:
        return 0.0
    # sliding energy of ref at each lag via cumsum (O(n), not O(n*seg))
    # window sum at lag k = sum(ref[k:k+seg.size]**2) = cs[seg.size-1+k] - cs[k-1]
    ref2 = np.cumsum((ref * ref).astype(np.float64))
    win = np.empty(lags + 1)
    win[0] = ref2[seg.size - 1]
    win[1:] = ref2[seg.size:seg.size + lags] - ref2[:lags]
    norms = np.sqrt(win)
    ok = norms > 1e-6
    if not ok.any():
        return 0.0
    scores = np.where(ok, np.abs(xcorr[:lags + 1]) / np.maximum(norms, 1e-6), 0.0)
    return float(np.max(scores) / seg_norm)


class EchoState:
    """Shared state: recently-played TTS audio + recent bot text."""

    def __init__(self, ref_secs: float = 4.0, mic_secs: float = 1.5):
        self.ref_secs = ref_secs
        self.mic_secs = mic_secs
        self.ref: list[tuple[float, np.ndarray]] = []   # (wallclock, 16k float32)
        self.mic: list[tuple[float, np.ndarray]] = []
        self.bot_texts: list[tuple[float, str]] = []    # (wallclock, normalized)

    # ---- played TTS audio (from the output tap) ----
    def add_played(self, samples_16k: np.ndarray) -> None:
        now = time.monotonic()
        if samples_16k.size:
            self.ref.append((now, samples_16k))
            self._trim(self.ref, self.ref_secs)

    def recent_ref(self, within_secs: float) -> np.ndarray | None:
        """Concatenated played audio, or None if nothing played recently."""
        now = time.monotonic()
        fresh = [(t, s) for t, s in self.ref if now - t <= within_secs]
        if not fresh:
            return None
        return np.concatenate([s for _, s in fresh])

    # ---- mic audio (from the detector itself) ----
    def add_mic(self, samples: np.ndarray) -> None:
        now = time.monotonic()
        self.mic.append((now, samples))
        self._trim(self.mic, self.mic_secs)

    def recent_mic(self, secs: float = 0.4) -> np.ndarray:
        now = time.monotonic()
        fresh = [s for t, s in self.mic if now - t <= secs]
        return np.concatenate(fresh) if fresh else np.zeros(0, np.float32)

    # ---- bot text (fed by the observer) ----
    def add_bot_text(self, text: str) -> None:
        norm = _normalize(text or "")
        if norm:
            self.bot_texts.append((time.monotonic(), norm))
            self._trim(self.bot_texts, 30.0)

    def recent_bot_text(self, within_secs: float) -> str:
        now = time.monotonic()
        fresh = [txt for t, txt in self.bot_texts if now - t <= within_secs]
        return " ".join(fresh)

    @staticmethod
    def _trim(buf: list, secs: float) -> None:
        if not buf:
            return
        cutoff = time.monotonic() - secs
        while buf and buf[0][0] < cutoff:
            buf.pop(0)


class TTSOutputTap(FrameProcessor):
    """Sits between the assistant aggregator and the output transport: records
    the TTS audio (resampled to 16 kHz) that is about to hit the speaker, and
    passes everything through untouched."""

    def __init__(self, state: EchoState, src_rate: int = 24000):
        super().__init__()
        self._state = state
        self._src_rate = src_rate

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.audio:
            pcm = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self._state.add_played(_resample_linear(pcm, self._src_rate, 16000))
        await self.push_frame(frame, direction)


class EchoDetectorProcessor(FrameProcessor):
    """See module docstring. Must be placed AFTER the VAD processor (it acts on
    UserStartedSpeakingFrame) and BEFORE the user context aggregator (it can
    veto the frame, suppressing the interruption/turn)."""

    def __init__(self, state: EchoState, corr_threshold: float = 0.4,
                 decision_window: float = 2.0, text_window: float = 4.0,
                 text_similarity: float = 0.65, decision_mic_secs: float = 0.4):
        super().__init__()
        self._state = state
        self._corr_threshold = corr_threshold
        self._decision_window = decision_window
        self._text_window = text_window
        self._text_similarity = text_similarity
        self._decision_mic_secs = decision_mic_secs
        self._dropping = False
        self._last_verdict = None

    # ---- burst classification ----
    def _is_echo(self) -> bool:
        ref = self._state.recent_ref(self._decision_window)
        if ref is None:
            return False
        mic = self._state.recent_mic(self._decision_mic_secs)
        score = _corr_score(mic, ref)
        self._last_verdict = round(score, 3)
        log.debug("echo correlation score=%.3f threshold=%.2f", score, self._corr_threshold)
        return score >= self._corr_threshold

    def _text_is_echo(self, text: str) -> bool:
        norm = _normalize(text or "")
        if len(norm) < 3:
            return False
        recent = self._state.recent_bot_text(self._text_window)
        if not recent:
            return False
        if norm in recent:
            return True
        # compare against the single most recent bot utterance for a sane ratio
        last = self._state.bot_texts[-1][1] if self._state.bot_texts else ""
        if last and difflib.SequenceMatcher(None, norm, last).ratio() >= self._text_similarity:
            return True
        return False

    # ---- frame processing ----
    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserAudioRawFrame) and frame.audio:
            pcm = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            rate = getattr(frame, "sample_rate", 16000) or 16000
            self._state.add_mic(_resample_linear(pcm, rate, 16000))
            if self._dropping:
                return  # suppress the whole classified-echo burst
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._state.recent_ref(self._decision_window) is not None:
                self._dropping = self._is_echo()
                if self._dropping:
                    log.info("echo burst suppressed (corr=%.3f)", self._last_verdict)
            else:
                self._dropping = False
                self._last_verdict = None
            if self._dropping:
                return  # veto: no interruption, no user turn
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if self._dropping or self._text_is_echo(str(getattr(frame, "text", ""))):
                if not self._dropping:
                    log.info("echo transcript suppressed (text match): %.60s",
                             getattr(frame, "text", ""))
                return  # never reaches the context / LLM
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._dropping = False

        await self.push_frame(frame, direction)
