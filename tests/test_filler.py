import asyncio

import pytest

from agent.filler import FillerSpeaker
from tests.mocks import MockLLMClient


def make_speaker(script, speak_log, thinking=0.05, tool=0.05):
    client = MockLLMClient(list(script))
    s = FillerSpeaker(speak=speak_log.append, client=client,
                      thinking_secs=thinking, tool_secs=tool)
    return s


@pytest.mark.asyncio
async def test_thinking_filler_fires_and_speaks_llm_text():
    spoken = []
    s = make_speaker([{"role": "assistant", "content": "Let me consider that for a moment."}],
                     spoken)
    s.on_llm_request(1, "what is quantum computing?")
    await asyncio.sleep(0.3)
    assert len(spoken) == 1
    assert "consider" in spoken[0]


@pytest.mark.asyncio
async def test_first_token_cancels_pending_filler():
    spoken = []
    s = make_speaker([{"role": "assistant", "content": "slow filler"}], spoken)
    s.on_llm_request(1, "hello")
    await asyncio.sleep(0.02)
    s.on_first_token(1)  # real response arrived before the threshold
    await asyncio.sleep(0.2)
    assert spoken == []


@pytest.mark.asyncio
async def test_tool_filler_uses_tool_phrase():
    spoken = []
    s = FillerSpeaker(speak=spoken.append,
                      client=MockLLMClient([{"role": "assistant", "content": ""}]),  # empty -> canned fallback
                      tool_secs=0.05)
    s.on_tool_running(1, "search_web")
    await asyncio.sleep(0.2)
    assert len(spoken) == 1 and "search" in spoken[0].lower()


@pytest.mark.asyncio
async def test_one_filler_per_turn():
    spoken = []
    s = make_speaker([{"role": "assistant", "content": "a"},
                      {"role": "assistant", "content": "b"}], spoken)
    s.on_llm_request(1, "q1")
    s.on_tool_running(1, "search_web")
    await asyncio.sleep(0.3)
    assert len(spoken) == 1


@pytest.mark.asyncio
async def test_fast_response_beats_filler():
    spoken = []
    s = make_speaker([{"role": "assistant", "content": "never spoken"}], spoken,
                     thinking=5.0)
    s.on_llm_request(1, "hi")
    await asyncio.sleep(0.1)
    s.on_bot_speech(1)   # real answer started speaking long before threshold
    await asyncio.sleep(0.2)
    assert spoken == []


@pytest.mark.asyncio
async def test_llm_error_falls_back_to_canned():
    class FailingClient:
        async def chat(self, *a, **kw):
            raise RuntimeError("down")
    spoken = []
    s = FillerSpeaker(speak=spoken.append, client=FailingClient(), thinking_secs=0.05)
    s.on_llm_request(1, "hi")
    await asyncio.sleep(0.25)
    assert len(spoken) == 1
    from agent.filler import CANNED_THINKING
    assert spoken[0] in CANNED_THINKING


@pytest.mark.asyncio
async def test_disabled_never_speaks():
    spoken = []
    s = FillerSpeaker(speak=spoken.append, client=MockLLMClient([]), thinking_secs=0.05,
                      enabled=False)
    s.on_llm_request(1, "hi")
    s.on_tool_running(1, "search_web")
    await asyncio.sleep(0.2)
    assert spoken == []