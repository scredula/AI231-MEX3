"""Echo suppression at the AUDIO level: keeps the mic open (barge-in works)
while preventing the speakers' TTS output from being transcribed back as user
speech.

Why audio-level, not frame-veto: pipecat 1.14's VADProcessor BROADCASTS
VADUserStartedSpeakingFrame and the user aggregator decides interruptions
internally — a processor sitting after the VAD cannot veto the decision. So the
detector must sit BEFORE the VAD/STT and simply not hand echo audio downstream.

Mechanism (EchoDetectorProcessor, placed between mic capture and STT):
- While no TTS played recently: audio passes straight through (zero added
  latency).
- While TTS played recently (echo-risk window): speech-level mic audio is held
  in a short decision buffer (not forwarded). Every ECHO_DECIDE_MS of buffered
  speech, it is cross-correlated against the recently played audio:
    * echo-like  -> keep holding/dropping (the burst is the bot hearing itself)
    * not echo   -> flush the whole buffer downstream and open the gate: a real
      barge-in reaches VAD/STT after at most ~240 ms of withheld audio
- Silence longer than ECHO_SILENCE_MS resets the state machine.

The reference signal is recorded by TTSOutputTap (between the assistant
aggregator and the output transport) and resampled to 16 kHz mono.

EchoTranscriptGuard (after the VAD, before the user aggregator) is the backstop:
transcripts arriving shortly after bot speech that fuzzy-match the bot's own
recent text are discarded — covers reverb tails that leak past the correlation.

The observer feeds bot text (LLMTextFrame deltas, TTSSpeakFrame) into the shared
EchoState via add_bot_text.
"""
import asyncio
import difflib
import logging
import re
import time

import numpy as np
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
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
    ref.size - seg.size. Returns the max Pearson-style coefficient: high for a
    delayed copy (echo), low for unrelated audio (real speech).
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
    xcorr = np.fft.irfft(fr * np.conj(fs), nfft)  # x[k] = sum ref[n+k]*seg[n]
    lags = ref.size - seg.size if max_lag is None else min(max_lag, ref.size - seg.size)
    if lags <= 0:
        return 0.0
    # sliding energy of ref at each lag via cumsum (O(n))
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

    def __init__(self, ref_secs: float = 4.0):
        self.ref_secs = ref_secs
        self.ref: list[tuple[float, np.ndarray]] = []   # (wallclock, 16k float32)
        self.bot_texts: list[tuple[float, str]] = []    # (wallclock, normalized)

    def add_played(self, samples_16k: np.ndarray) -> None:
        now = time.monotonic()
        if samples_16k.size:
            self.ref.append((now, samples_16k))
            self._trim(self.ref, self.ref_secs)

    def recent_ref(self, within_secs: float) -> np.ndarray | None:
        now = time.monotonic()
        fresh = [(t, s) for t, s in self.ref if now - t <= within_secs]
        if not fresh:
            return None
        return np.concatenate([s for _, s in fresh])

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
        self._default_rate = src_rate
        self._seen = 0

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        # the TTS pushes TTSAudioRawFrame (OutputAudioRawFrame subclass); accept
        # any output audio frame and trust the frame's own sample rate
        if isinstance(frame, OutputAudioRawFrame) and frame.audio:
            pcm = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            rate = getattr(frame, "sample_rate", None) or self._default_rate
            self._state.add_played(_resample_linear(pcm, rate, 16000))
            self._seen += 1
            if self._seen % 100 == 0:
                log.debug("tts tap: %d audio frames captured", self._seen)
        await self.push_frame(frame, direction)


class EchoDetectorProcessor(FrameProcessor):
    """Audio-level echo suppression. Place between mic capture and STT."""

    def __init__(self, state: EchoState, corr_threshold: float = 0.4,
                 decision_window: float = 2.5, decide_ms: int = 240,
                 silence_rms: float = 0.006, silence_ms: int = 150):
        super().__init__()
        self._state = state
        self._threshold = corr_threshold
        self._decision_window = decision_window
        self._decide_samples = int(decide_ms * 16)       # ms -> samples @16k
        self._silence_rms = silence_rms
        self._silence_secs = silence_ms / 1000.0
        self._dropping = False            # currently suppressing (echo confirmed)
        self._holding = False             # buffering while undecided
        self._buf: list[tuple[object, np.ndarray]] = []  # (original frame, 16k copy)
        self._silence_run = 0.0           # accumulated AUDIO seconds of silence
        self._mic_frames = 0

    # ---- helpers ----
    def _flush(self) -> None:
        """Push every held frame downstream, in order."""
        buf, self._buf = self._buf, []
        self._holding = False
        self._dropping = False
        async def _push_all():
            for frame, _ in buf:
                await self.push_frame(frame)
        asyncio.get_event_loop().create_task(_push_all())

    def _discard(self) -> None:
        self._buf.clear()
        self._holding = False
        self._dropping = True

    # ---- frame processing ----
    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, InputAudioRawFrame) or not frame.audio:
            await self.push_frame(frame, direction)
            return
        self._mic_frames += 1
        if self._mic_frames % 500 == 0:
            ref = self._state.recent_ref(self._decision_window)
            log.debug("echo detector alive: mic frames=%d, ref %s",
                      self._mic_frames,
                      f"{len(self._state.ref)} chunks" if ref is not None else "EMPTY")

        ref = self._state.recent_ref(self._decision_window)
        if ref is None:
            # no recent TTS — everything passes, reset any held state
            if self._buf:
                self._flush()
            self._dropping = False
            await self.push_frame(frame, direction)
            return

        # echo-risk window active
        pcm = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
        rate = getattr(frame, "sample_rate", 16000) or 16000
        samples16k = _resample_linear(pcm, rate, 16000)
        rms = float(np.sqrt((samples16k * samples16k).mean())) if samples16k.size else 0.0
        now = time.monotonic()

        if rms < self._silence_rms:
            # accumulate AUDIO time (robust to bursty frame delivery), not wallclock
            self._silence_run += samples16k.size / 16000.0
            if self._silence_run >= self._silence_secs:
                # sustained silence: the echo burst is over — reset the state machine
                if self._holding and self._buf:
                    self._flush()          # let STT/VAD see it; the guard cleans up
                self._dropping = False
            await self.push_frame(frame, direction)   # silence is harmless
            return
        self._silence_run = 0.0

        if self._dropping:
            return  # echo confirmed for this burst — keep suppressing

        self._holding = True
        self._buf.append((frame, samples16k))
        held = np.concatenate([s for _, s in self._buf])

        if held.size >= self._decide_samples:
            score = _corr_score(held, ref)
            log.debug("echo decision score=%.3f threshold=%.2f", score, self._threshold)
            if score >= self._threshold:
                log.info("echo suppressed (corr=%.3f, %.0f ms held)",
                         score, 1000 * held.size / 16000)
                self._discard()
            else:
                log.info("real speech confirmed (corr=%.3f) — flushing %.0f ms",
                         score, 1000 * held.size / 16000)
                self._flush()

        if self._holding or self._dropping:
            return  # held frames are flushed/rejected as a block
        await self.push_frame(frame, direction)


class EchoTranscriptGuard(FrameProcessor):
    """Backstop placed AFTER the VAD and BEFORE the user aggregator: drops
    transcripts that fuzzy-match the bot's own recent speech (reverb tails the
    audio-level detector let through)."""

    def __init__(self, state: EchoState, text_window: float = 4.0,
                 text_similarity: float = 0.65):
        super().__init__()
        self._state = state
        self._text_window = text_window
        self._text_similarity = text_similarity

    def _text_is_echo(self, text: str) -> bool:
        norm = _normalize(text or "")
        if not norm:
            return False
        recent = self._state.recent_bot_text(self._text_window)
        if not recent:
            return False
        if len(norm) < 6:
            # short fragments ("i", "yep", "take your time") must match a whole
            # word the bot actually said — protects genuine "yes"/"no" replies
            return bool(re.search(rf"\b{re.escape(norm)}\b", recent))
        if norm in recent:
            return True
        last = self._state.bot_texts[-1][1] if self._state.bot_texts else ""
        return bool(last and
                    difflib.SequenceMatcher(None, norm, last).ratio() >= self._text_similarity)

    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if self._text_is_echo(str(getattr(frame, "text", ""))):
                log.info("echo transcript suppressed: %.60s", getattr(frame, "text", ""))
                return
        await self.push_frame(frame, direction)