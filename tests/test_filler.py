import asyncio

import pytest

from agent.filler import CANNED_THINKING, FillerSpeaker
from tests.mocks import MockLLMClient


async def collect(s, seconds=0.4, arm=None):
    """Install a spoken-phrase collector BEFORE arming, then run `arm` (sync or async)
    and wait."""
    spoken = []

    async def speak(t):
        spoken.append(t)
    s.speak = speak
    if arm:
        res = arm()
        if asyncio.iscoroutine(res):
            await res
    await asyncio.sleep(seconds)
    return spoken


def make(always, script=None, **kw):
    kw.setdefault("tool_secs", 0.05)
    return FillerSpeaker(speak=lambda t: asyncio.sleep(0),
                         client=MockLLMClient(list(script or [])),
                         always=always, **kw)


# ---------------- always mode (default) ----------------

@pytest.mark.asyncio
async def test_always_filler_fires_immediately_after_transcript():
    s = make(True, [{"role": "assistant", "content": "Let me look that up for you."}])
    spoken = await collect(s, arm=lambda: s.on_transcript(1, "What is Florida's record?"))
    assert len(spoken) == 1 and "look" in spoken[0].lower()


@pytest.mark.asyncio
async def test_always_filler_not_cancelled_by_first_token():
    s = make(True, [{"role": "assistant", "content": "One moment."}])
    spoken = await collect(s, 0.3, arm=lambda: (s.on_transcript(1, "hello"),
                                                s.on_first_token(1)))
    assert len(spoken) == 1  # real response starts — immediate filler still speaks


@pytest.mark.asyncio
async def test_always_no_llm_request_filler():
    s = make(True, [{"role": "assistant", "content": "x"}])
    spoken = await collect(s, arm=lambda: s.on_llm_request(1, "hi"))
    assert spoken == []  # armed at the transcript, not per LLM call


@pytest.mark.asyncio
async def test_always_tool_filler_canned_after_immediate():
    s = make(True, [{"role": "assistant", "content": "immediate"}], tool_secs=0.02)

    async def arm():
        s.on_transcript(1, "search for news")
        await asyncio.sleep(0.1)   # immediate filler speaks
        s.on_tool_running(1, "search_web")
    spoken = await collect(s, 0.3, arm)
    assert len(spoken) == 2 and "search" in spoken[1].lower()


@pytest.mark.asyncio
async def test_always_still_working_after_utterance():
    s = make(True, [{"role": "assistant", "content": "immediate"}], tool_secs=0.02)

    async def arm():
        s.on_transcript(1, "do a long task")
        await asyncio.sleep(0.1)
        s.on_bot_speech_end(1, work_pending=True)
    spoken = await collect(s, 0.3, arm)
    assert len(spoken) == 2 and "working" in spoken[1].lower()


@pytest.mark.asyncio
async def test_cap_limits_fillers_per_turn():
    s = make(True, [{"role": "assistant", "content": "f"}], max_per_turn=2,
             tool_secs=0.02)

    async def arm():
        s.on_transcript(1, "q")
        await asyncio.sleep(0.1)
        s.on_bot_speech_end(1, work_pending=True)
        await asyncio.sleep(0.1)
        s.on_bot_speech_end(1, work_pending=True)
        await asyncio.sleep(0.1)
        s.on_bot_speech_end(1, work_pending=True)  # over the cap
    spoken = await collect(s, 0.3, arm)
    assert len(spoken) == 2


# ---------------- threshold mode (FILLER_ALWAYS=0) ----------------

@pytest.mark.asyncio
async def test_threshold_thinking_filler_fires():
    s = make(False, [{"role": "assistant", "content": "Let me consider that."}],
             thinking_secs=0.05)
    spoken = await collect(s, arm=lambda: s.on_llm_request(1, "what is quantum computing?"))
    assert len(spoken) == 1 and "consider" in spoken[0].lower()


@pytest.mark.asyncio
async def test_threshold_first_token_cancels_thinking():
    s = make(False, [{"role": "assistant", "content": "slow filler"}],
             thinking_secs=5.0)
    spoken = await collect(s, 0.2, arm=lambda: (s.on_llm_request(1, "hello"),
                                                s.on_first_token(1)))
    assert spoken == []


@pytest.mark.asyncio
async def test_threshold_transcript_alone_does_not_speak():
    s = make(False, [{"role": "assistant", "content": "x"}], thinking_secs=0.05)
    spoken = await collect(s, arm=lambda: s.on_transcript(1, "hello"))
    assert spoken == []  # threshold mode: the LLM request is the trigger, not STT


@pytest.mark.asyncio
async def test_threshold_tool_filler_canned_when_no_llm_text():
    s = make(False, [{"role": "assistant", "content": ""}], tool_secs=0.05)
    spoken = await collect(s, 0.3, arm=lambda: (s.on_llm_request(1, "search news"),
                                                s.on_first_token(1),
                                                s.on_tool_running(1, "search_web")))
    assert len(spoken) == 1 and "search" in spoken[0].lower()


# ---------------- shared behavior ----------------

@pytest.mark.asyncio
async def test_filler_model_error_falls_back_to_canned():
    class Failing:
        async def chat(self, *a, **kw):
            raise RuntimeError("down")
    s = FillerSpeaker(speak=lambda t: asyncio.sleep(0), client=Failing(),
                      always=True, tool_secs=0.02)
    spoken = await collect(s, 0.3, arm=lambda: s.on_transcript(1, "hi"))
    assert len(spoken) == 1 and spoken[0] in CANNED_THINKING


@pytest.mark.asyncio
async def test_disabled_never_speaks():
    s = make(True, [{"role": "assistant", "content": "x"}], enabled=False)
    spoken = await collect(s, arm=lambda: s.on_transcript(1, "hi"))
    assert spoken == []


def test_filler_system_prompt_caps_words():
    from agent.filler import FILLER_SYSTEM
    assert "8 words" in FILLER_SYSTEM