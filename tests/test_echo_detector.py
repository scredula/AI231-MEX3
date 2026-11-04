import asyncio

import numpy as np
import pytest
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent.echo_detector import (
    EchoDetectorProcessor,
    EchoState,
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


async def feed(proc, frames, collector, settle=0.02):
    proc.push_frame = lambda f, d=None: collector.process_frame(f, FrameDirection.DOWNSTREAM)
    for f in frames:
        await proc.process_frame(f, FrameDirection.DOWNSTREAM)
    await asyncio.sleep(settle)


def make_detector(**kw):
    state = EchoState(ref_secs=4.0, mic_secs=1.5)
    return EchoDetectorProcessor(state, **kw), state


# ---------- correlation math ----------

def test_corr_high_for_echo_slice():
    # physical echo: the mic window is a late slice of the played audio (the mic
    # buffer trails the reference buffer), attenuated by the room
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


# ---------- end-to-end burst suppression ----------

@pytest.mark.asyncio
async def test_echo_burst_is_suppressed():
    det, state = make_detector(corr_threshold=0.35)
    col = Collector()
    rng = np.random.default_rng(3)

    # 1) bot plays 2 s of audio (the tap records it as the echo reference)
    ref = (rng.standard_normal(RATE * 2) * 0.3).astype(np.float32)
    for i in range(20):  # 100 ms chunks of 16k audio, upsampled to 24k for the tap
        chunk = _resample_linear(ref[i * 1600:(i + 1) * 1600], RATE, 24000)
        await TTSOutputTap(state).process_frame(
            TTSAudioRawFrame(audio=pcm16(chunk, 24000), sample_rate=24000, num_channels=1),
            FrameDirection.DOWNSTREAM)

    # 2) the mic hears a delayed copy of it (echo)
    echo = np.concatenate([np.zeros(int(0.2 * RATE), np.float32),
                           ref[int(0.2 * RATE):int(0.6 * RATE)] * 0.8])
    for i in range(64):  # 400 ms of echo, 10 ms chunks? -> use 100ms chunks
        det._state.add_mic(echo[i * 1600:(i + 1) * 1600])

    # 3) VAD fires speech start -> detector must classify as echo and veto
    await feed(det, [UserStartedSpeakingFrame()], col)
    # 4) transcripts from that burst must be dropped
    await feed(det, [TranscriptionFrame("I can hear you loud and clear", "", "0")], col)
    types = [type(f).__name__ for f in col.frames]
    assert "UserStartedSpeakingFrame" not in types    # vetoed — no interruption
    assert "TranscriptionFrame" not in types          # no echo transcript reaches context


@pytest.mark.asyncio
async def test_real_user_speech_passes_and_barge_in_works():
    det, state = make_detector(corr_threshold=0.35)
    col = Collector()
    rng = np.random.default_rng(5)

    ref = (rng.standard_normal(RATE * 2) * 0.3).astype(np.float32)
    state.add_played(ref)   # bot just played something

    user = (rng.standard_normal(int(0.4 * RATE)) * 0.3).astype(np.float32)  # unrelated audio
    state.add_mic(user)

    await feed(det, [UserStartedSpeakingFrame()], col)
    await feed(det, [TranscriptionFrame("actually stop the music", "", "0")], col)
    types = [type(f).__name__ for f in col.frames]
    assert "UserStartedSpeakingFrame" in types    # barge-in preserved
    assert "TranscriptionFrame" in types          # real speech reaches the LLM


@pytest.mark.asyncio
async def test_speech_long_after_tts_passes_uncorrelated():
    det, state = make_detector()
    col = Collector()
    await feed(det, [UserStartedSpeakingFrame()], col)
    await feed(det, [TranscriptionFrame("what time is it", "", "0")], col)
    types = [type(f).__name__ for f in col.frames]
    assert "UserStartedSpeakingFrame" in types and "TranscriptionFrame" in types


# ---------- transcript backstop ----------

@pytest.mark.asyncio
async def test_backstop_drops_transcript_matching_bot_text():
    det, state = make_detector()
    col = Collector()
    state.add_bot_text("Yes, I can hear you loud and clear — what can I help you with?")

    await feed(det, [TranscriptionFrame("I can hear you loud and clear", "", "0")], col)
    assert all(type(f).__name__ != "TranscriptionFrame" for f in col.frames)


@pytest.mark.asyncio
async def test_backstop_keeps_genuine_reply():
    det, state = make_detector()
    col = Collector()
    state.add_bot_text("Can you hear me okay?")

    await feed(det, [TranscriptionFrame("Yes you sound great", "", "0")], col)
    assert any(type(f).__name__ == "TranscriptionFrame" for f in col.frames)


@pytest.mark.asyncio
async def test_interim_transcript_dropped_during_echo_burst():
    det, state = make_detector(corr_threshold=0.35)
    col = Collector()
    state.add_played(np.random.default_rng(2).standard_normal(RATE).astype(np.float32))
    state.add_mic(state.recent_ref(2.0)[-6400:])  # mic hears the last 0.4s of playback
    await feed(det, [UserStartedSpeakingFrame()], col)
    await feed(det, [InterimTranscriptionFrame("I can hear", "", "0")], col)
    assert all(type(f).__name__ != "InterimTranscriptionFrame" for f in col.frames)


@pytest.mark.asyncio
async def test_burst_state_resets_on_speech_stop():
    det, state = make_detector(corr_threshold=0.35)
    col = Collector()
    state.add_played(np.random.default_rng(2).standard_normal(RATE).astype(np.float32))
    state.add_mic(state.recent_ref(2.0)[-6400:])
    await feed(det, [UserStartedSpeakingFrame()], col)
    assert det._dropping is True
    await feed(det, [UserStoppedSpeakingFrame()], col)
    assert det._dropping is False
