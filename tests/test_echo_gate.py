import asyncio

import pytest
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, UserAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

from agent.pipeline import EchoGateProcessor


class Sink:
    """Collects frames pushed by the gate."""

    def __init__(self):
        self.frames = []

    def audio_frames(self):
        return [f for f in self.frames if isinstance(f, UserAudioRawFrame)]


def wire_push(gate, sink):
    """Replace the gate's push_frame with an async recorder (instance attr)."""

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        sink.frames.append(frame)
    gate.push_frame = push


def audio() -> UserAudioRawFrame:
    return UserAudioRawFrame(audio=b"\x00" * 100, sample_rate=16000, num_channels=1)


@pytest.mark.asyncio
async def test_gate_drops_audio_while_bot_speaks():
    gate = EchoGateProcessor(tail_secs=0.05)
    sink = Sink()
    wire_push(gate, sink)

    # the observer calls on_bot_speech_start when it sees the TTS start
    gate.on_bot_speech_start()
    for f in [audio(), audio()]:
        await gate.process_frame(f, FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)
    assert sink.audio_frames() == []    # mic audio dropped while speaking


@pytest.mark.asyncio
async def test_gate_passes_audio_when_idle():
    gate = EchoGateProcessor(tail_secs=0.05)
    sink = Sink()
    wire_push(gate, sink)

    await gate.process_frame(audio(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)
    assert len(sink.audio_frames()) == 1


@pytest.mark.asyncio
async def test_gate_reopens_after_tail():
    gate = EchoGateProcessor(tail_secs=0.05)
    sink = Sink()
    wire_push(gate, sink)

    gate.on_bot_speech_start()
    gate.on_bot_speech_end()            # schedules reopen after the 0.05s tail
    await asyncio.sleep(0.2)
    await gate.process_frame(audio(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.05)
    assert len(sink.audio_frames()) == 1  # audio passes again after the tail


@pytest.mark.asyncio
async def test_gate_new_speech_cancels_pending_reopen():
    gate = EchoGateProcessor(tail_secs=0.05)
    sink = Sink()
    wire_push(gate, sink)

    gate.on_bot_speech_start()
    gate.on_bot_speech_end()            # schedules reopen in 0.05s
    await asyncio.sleep(0.02)
    gate.on_bot_speech_start()          # new utterance cancels the pending reopen
    await asyncio.sleep(0.1)
    assert gate._muted is True          # still muted — echo would still be audible
    await gate.process_frame(audio(), FrameDirection.DOWNSTREAM)
    assert sink.audio_frames() == []    # audio still dropped