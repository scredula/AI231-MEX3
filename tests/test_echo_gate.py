import asyncio

import pytest
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, UserAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

from agent.pipeline import EchoGateProcessor


class Sink:
    """Collects frames pushed by the gate."""

    def __init__(self):
        self.frames = []

    async def process_frame(self, frame, direction):
        self.frames.append(frame)


def wire_push(gate, sink):
    """Replace the gate's push_frame with an async recorder (instance attr)."""

    async def push(frame, direction=FrameDirection.DOWNSTREAM):
        sink.frames.append(frame)
    gate.push_frame = push


def audio() -> UserAudioRawFrame:
    return UserAudioRawFrame(audio=b"\x00" * 100, sample_rate=16000, num_channels=1)


async def run_gate(gate, seq, sink, wait=0.05):
    for f in seq:
        await gate.process_frame(f, FrameDirection.DOWNSTREAM)
    await asyncio.sleep(wait)


@pytest.mark.asyncio
async def test_gate_drops_audio_while_bot_speaks():
    gate = EchoGateProcessor(tail_secs=0.05)
    sink = Sink()
    wire_push(gate, sink)

    await run_gate(gate, [BotStartedSpeakingFrame(), audio(), audio()], sink)
    assert sink.frames == []            # all mic audio dropped while speaking


@pytest.mark.asyncio
async def test_gate_passes_audio_when_idle():
    gate = EchoGateProcessor(tail_secs=0.05)
    sink = Sink()
    wire_push(gate, sink)

    await run_gate(gate, [audio()], sink)
    assert len(sink.frames) == 1


@pytest.mark.asyncio
async def test_gate_reopens_after_tail():
    gate = EchoGateProcessor(tail_secs=0.05)
    sink = Sink()
    wire_push(gate, sink)

    await run_gate(gate, [BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), audio()],
                   sink, wait=0.2)  # wait past the 0.05s tail
    assert len(sink.frames) == 1        # audio passes again after the tail


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