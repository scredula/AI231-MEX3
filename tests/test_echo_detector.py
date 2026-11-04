import asyncio

import numpy as np
import pytest
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.echo_detector import (
    EchoDetectorProcessor,
    EchoState,
    EchoTranscriptGuard,
    TTSOutputTap,
    _corr_score,
    _resample_linear,
)

RATE = 16000


def pcm16(samples_f32, rate=RATE):
    return (np.clip(samples_f32, -1, 1) * 32767).astype("<i2").tobytes()


def audio_frame(samples_f32, rate=RATE):
    return UserAudioRawFrame(audio=pcm16(samples_f32, rate), sample_rate=rate, num_channels=1)


class Collector(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def run_through(proc, frames, sink, settle=0.08):
    """Route frames through `proc` into `sink` (proc.push_frame -> sink)."""
    async def push(f, d=None):
        await sink.process_frame(f, FrameDirection.DOWNSTREAM)
    proc.push_frame = push
    for f in frames:
        await proc.process_frame(f, FrameDirection.DOWNSTREAM)
    await asyncio.sleep(settle)


def play_ref(state, secs=2.0, seed=3):
    """Record `secs` of fake TTS into the state (as the output tap would)."""
    rng = np.random.default_rng(seed)
    ref = (rng.standard_normal(int(RATE * secs)) * 0.3).astype(np.float32)
    tap = TTSOutputTap(state)
    for i in range(0, len(ref), 1600):  # 100 ms @16k, upsampled to 24k like real TTS
        chunk = _resample_linear(ref[i:i + 1600], RATE, 24000)
        asyncio.get_event_loop().create_task(tap.process_frame(
            TTSAudioRawFrame(audio=pcm16(chunk, 24000), sample_rate=24000, num_channels=1),
            FrameDirection.DOWNSTREAM))
    return ref


async def play_ref_async(state, secs=2.0, seed=3):
    rng = np.random.default_rng(seed)
    ref = (rng.standard_normal(int(RATE * secs)) * 0.3).astype(np.float32)
    tap = TTSOutputTap(state)
    for i in range(0, len(ref), 1600):
        chunk = _resample_linear(ref[i:i + 1600], RATE, 24000)
        await tap.process_frame(
            TTSAudioRawFrame(audio=pcm16(chunk, 24000), sample_rate=24000, num_channels=1),
            FrameDirection.DOWNSTREAM)
    return ref


def make_detector(**kw):
    state = EchoState()
    kw.setdefault("corr_threshold", 0.35)
    return EchoDetectorProcessor(state, **kw), state


# ---------- correlation math ----------

def test_corr_high_for_echo_slice():
    # physical echo: the mic window is a late slice of the played audio
    rng = np.random.default_rng(7)
    ref = rng.standard_normal(RATE * 2).astype(np.float32) * 0.2
    seg = ref[2000:2000 + 4000] * 0.7
    assert _corr_score(seg, ref, int(0.7 * RATE)) > 0.9


def test_corr_low_for_unrelated_audio():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(RATE * 2).astype(np.float32) * 0.2
    seg = rng.standard_normal(4000).astype(np.float32) * 0.2
    assert _corr_score(seg, ref, int(0.7 * RATE)) < 0.2


def test_resample_changes_length():
    x = np.ones(24000, np.float32)
    assert _resample_linear(x, 24000, 16000).size == 16000


# ---------- audio-level suppression ----------

@pytest.mark.asyncio
async def test_echo_audio_never_reaches_stt():
    det, state = make_detector()
    sink = Collector()
    ref = await play_ref_async(state)

    # mic hears a delayed, attenuated copy of what was played
    echo = np.concatenate([np.zeros(int(0.2 * RATE), np.float32),
                           ref[int(0.2 * RATE):int(0.7 * RATE)] * 0.8])
    frames = [audio_frame(echo[i:i + 1600]) for i in range(0, len(echo), 1600)]
    await run_through(det, frames, sink)

    def rms(f):
        x = np.frombuffer(f.audio, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt((x * x).mean())) if x.size else 0.0
    voiced = [f for f in sink.frames if isinstance(f, UserAudioRawFrame) and rms(f) > 0.006]
    assert voiced == []                # no speech-level echo audio reached downstream
    assert det._dropping is True


@pytest.mark.asyncio
async def test_real_speech_flashes_through_and_gate_opens():
    det, state = make_detector()
    sink = Collector()
    await play_ref_async(state, seed=5)

    user = (np.random.default_rng(6).standard_normal(int(0.5 * RATE)) * 0.3).astype(np.float32)
    frames = [audio_frame(user[i:i + 1600]) for i in range(0, len(user), 1600)]
    await run_through(det, frames, sink)

    passed = [f for f in sink.frames if isinstance(f, UserAudioRawFrame)]
    assert len(passed) >= 3            # held ≤240 ms then flushed, then free flow
    total = sum(f.audio and len(f.audio) // 2 for f in passed)
    assert total >= int(0.4 * RATE)    # ~all of the utterance reached downstream
    assert det._dropping is False


@pytest.mark.asyncio
async def test_no_recent_tts_means_zero_interference():
    det, state = make_detector()
    sink = Collector()
    user = (np.random.default_rng(8).standard_normal(int(0.3 * RATE)) * 0.3).astype(np.float32)
    frames = [audio_frame(user[i:i + 1600]) for i in range(0, len(user), 1600)]
    await run_through(det, frames, sink)
    assert len([f for f in sink.frames if isinstance(f, UserAudioRawFrame)]) == len(frames)


@pytest.mark.asyncio
async def test_silence_resets_dropping_state():
    det, state = make_detector()
    sink = Collector()
    ref = await play_ref_async(state)
    echo = ref * 0.8
    await run_through(det, [audio_frame(echo[i:i + 1600]) for i in range(0, len(echo), 1600)],
                      sink)
    assert det._dropping is True

    silence = np.zeros(int(0.3 * RATE), np.float32)  # 300 ms > silence_ms
    await run_through(det, [audio_frame(silence[i:i + 1600])
                            for i in range(0, len(silence), 1600)], sink)
    assert det._dropping is False


# ---------- transcript backstop ----------

@pytest.mark.asyncio
async def test_guard_drops_transcript_matching_bot_text():
    state = EchoState()
    state.add_bot_text("Yes, I can hear you loud and clear — what can I help you with?")
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("I can hear you loud and clear", "", "0")], sink)
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_keeps_genuine_reply():
    state = EchoState()
    state.add_bot_text("Can you hear me okay?")
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("Yes you sound great", "", "0")], sink)
    assert any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_drops_interim_echo():
    state = EchoState()
    state.add_bot_text("I'm listening — go ahead.")
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [InterimTranscriptionFrame("I'm listening", "", "0")], sink)
    assert not any(isinstance(f, InterimTranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_ignores_when_bot_quiet():
    state = EchoState()
    import time as _t
    state.bot_texts = [(_t.monotonic() - 10.0, "old sentence")]  # well outside the window
    guard = EchoTranscriptGuard(state, text_window=1.0)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("old sentence", "", "0")], sink)
    assert any(isinstance(f, TranscriptionFrame) for f in sink.frames)