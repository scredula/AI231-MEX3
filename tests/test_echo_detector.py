import asyncio

import numpy as np
import pytest
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
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
    # the local transport pushes plain InputAudioRawFrame (not the subclass)
    return InputAudioRawFrame(audio=pcm16(samples_f32, rate), sample_rate=rate, num_channels=1)


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
    voiced = [f for f in sink.frames if isinstance(f, InputAudioRawFrame) and rms(f) > 0.006]
    assert voiced == []                # no speech-level echo audio reached downstream
    assert det._dropping is True


@pytest.mark.asyncio
async def test_real_speech_flashes_through_and_gate_opens():
    det, state = make_detector()
    sink = Collector()
    await play_ref_async(state, seed=5)

    user = (np.random.default_rng(6).standard_normal(int(0.5 * RATE)) * 0.3).astype(np.float32)
    frames = [audio_frame(user[i:i + 1600]) for i in range(0, len(user), 1600)]
    silence = np.zeros(int(0.3 * RATE), np.float32)   # burst ends -> held tail must flush
    frames += [audio_frame(silence[i:i + 1600]) for i in range(0, len(silence), 1600)]
    await run_through(det, frames, sink)

    passed = [f for f in sink.frames if isinstance(f, InputAudioRawFrame)]
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
    assert len([f for f in sink.frames if isinstance(f, InputAudioRawFrame)]) == len(frames)


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
    state.last_playback_end = __import__("time").monotonic()
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("I can hear you loud and clear", "", "0")], sink)
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_keeps_genuine_reply():
    state = EchoState()
    state.add_bot_text("Can you hear me okay?")
    state.last_playback_end = __import__("time").monotonic()
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("Yes you sound great", "", "0")], sink)
    assert any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_drops_interim_echo():
    state = EchoState()
    state.add_bot_text("I'm listening — go ahead.")
    state.last_playback_end = __import__("time").monotonic()
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [InterimTranscriptionFrame("I'm listening", "", "0")], sink)
    assert not any(isinstance(f, InterimTranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_ignores_when_bot_quiet():
    state = EchoState()
    import time as _t
    state.bot_texts = [(_t.monotonic() - 10.0, "old sentence")]  # well outside the window
    state.last_playback_end = _t.monotonic() - 5.0               # playback ended long ago
    guard = EchoTranscriptGuard(state, text_window=1.0)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("old sentence", "", "0")], sink)
    assert any(isinstance(f, TranscriptionFrame) for f in sink.frames)

@pytest.mark.asyncio
async def test_guard_drops_short_fragments_of_bot_speech():
    state = EchoState()
    state.add_bot_text("Yep, I'm here and listening — what can I do for you?")
    state.last_playback_end = __import__("time").monotonic()
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("I", "", "0"),
                              TranscriptionFrame("Yep.", "", "0")], sink)
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_keeps_genuine_short_reply_not_in_bot_text():
    state = EchoState()
    state.add_bot_text("What would you like me to do?")
    state.last_playback_end = __import__("time").monotonic()
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("no", "", "0")], sink)
    assert any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_drops_stutter_echo_right_after_bot_speech():
    state = EchoState()
    state.add_bot_text("It's just past midnight in New York — 12:08 AM on Wednesday.")
    state.last_playback_end = __import__("time").monotonic()
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("It's just it's just", "", "0")], sink)
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_guard_keeps_late_genuine_reuse_of_bot_words():
    import time as _t
    state = EchoState()
    state.add_bot_text("Would you like me to publish it to GitHub?")
    state.bot_texts[-1] = (_t.monotonic() - 3.0, state.bot_texts[-1][1])  # 3s ago
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("i would say yes please", "", "0")], sink)
    assert any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_quiet_echo_still_suppressed():
    """Real-room failure mode: the echo is quieter than the old RMS floor."""
    det, state = make_detector()
    sink = Collector()
    ref = await play_ref_async(state)
    quiet_echo = ref * 0.05                     # well below the old 0.006 floor
    frames = [audio_frame(quiet_echo[i:i + 1600]) for i in range(0, len(quiet_echo), 1600)]
    await run_through(det, frames, sink)
    voiced = [f for f in sink.frames if isinstance(f, InputAudioRawFrame)
              and np.sqrt((np.frombuffer(f.audio, np.int16).astype(np.float32) / 32768.0) ** 2 .mean()) > 0.0015]
    assert voiced == []
    assert det._dropping is True


@pytest.mark.asyncio
async def test_embargo_drops_garbled_echo_during_active_echo():
    state = EchoState()
    state.add_bot_text("It's just after midnight in New York — 12:16 AM on Wednesday, November 4th.")
    state.last_suppressed = __import__("time").monotonic() - 0.5   # echo active NOW
    state.last_playback_end = __import__("time").monotonic()
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("Six to sixteen AM on Wednesday.", "", "0")], sink)
    assert not any(isinstance(f, TranscriptionFrame) for f in sink.frames)


@pytest.mark.asyncio
async def test_embargo_expires_genuine_followup_passes():
    import time as _t
    state = EchoState()
    state.add_bot_text("It's just after midnight in New York — 12:16 AM on Wednesday.")
    state.last_suppressed = _t.monotonic() - 5.0   # no recent suppression
    state.last_playback_end = _t.monotonic() - 4.0  # outside the 3s window anyway
    guard = EchoTranscriptGuard(state)
    sink = Collector()
    await run_through(guard, [TranscriptionFrame("what about Manila?", "", "0")], sink)
    assert any(isinstance(f, TranscriptionFrame) for f in sink.frames)
