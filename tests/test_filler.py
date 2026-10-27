import asyncio

import pytest

from agent.filler import CANNED_THINKING, FillerSpeaker, TOOL_CANNED


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


def make(always, **kw):
    kw.setdefault("tool_secs", 0.05)
    kw.setdefault("delay_secs", 0.05)  # tests use a short filler delay (prod: 3s)
    return FillerSpeaker(speak=lambda t: asyncio.sleep(0), always=always, **kw)


# ---------------- always mode (default) ----------------

@pytest.mark.asyncio
async def test_always_filler_fires_after_delay_if_no_llm_response():
    s = make(True)
    spoken = await collect(s, arm=lambda: s.on_transcript(1, "What is Florida's record?"))
    assert len(spoken) == 1 and spoken[0] in CANNED_THINKING


@pytest.mark.asyncio
async def test_always_filler_is_canned_no_model():
    s = make(True)
    spoken = await collect(s, arm=lambda: s.on_transcript(1, "hello"))
    assert spoken[0] in CANNED_THINKING  # never anything model-generated


@pytest.mark.asyncio
async def test_always_filler_cancelled_by_first_token():
    s = make(True, delay_secs=5.0)  # would fire at 5s
    spoken = await collect(s, 0.3, arm=lambda: (s.on_transcript(1, "hello"),
                                                s.on_first_token(1)))
    assert spoken == []  # LLM responded quickly — no filler


@pytest.mark.asyncio
async def test_always_filler_cancelled_by_bot_speech():
    s = make(True, delay_secs=5.0)
    spoken = await collect(s, 0.3, arm=lambda: (s.on_transcript(1, "hello"),
                                                s.on_bot_speech(1)))
    assert spoken == []  # bot already speaking — no filler


@pytest.mark.asyncio
async def test_always_no_llm_request_filler():
    s = make(True)
    spoken = await collect(s, arm=lambda: s.on_llm_request(1, "hi"))
    assert spoken == []  # armed at the transcript, not per LLM call


@pytest.mark.asyncio
async def test_always_no_tool_filler_after_immediate():
    s = make(True, tool_secs=0.02)

    async def arm():
        s.on_transcript(1, "search for news")
        await asyncio.sleep(0.1)   # filler fires after delay_secs=0.05
        s.on_tool_running(1, "search_web")   # always-mode: must be skipped
    spoken = await collect(s, 0.3, arm)
    assert len(spoken) == 1  # only the immediate filler; no mid-turn chatter


@pytest.mark.asyncio
async def test_always_still_working_after_utterance():
    s = make(True, tool_secs=0.02)

    async def arm():
        s.on_transcript(1, "do a long task")
        await asyncio.sleep(0.1)   # filler fires after delay_secs=0.05
        s.on_bot_speech_end(1, work_pending=True)
    spoken = await collect(s, 0.3, arm)
    assert len(spoken) == 2 and spoken[1] == TOOL_CANNED["_still_working"]


@pytest.mark.asyncio
async def test_cap_limits_fillers_per_turn():
    s = make(True, max_per_turn=2, tool_secs=0.02)

    async def arm():
        s.on_transcript(1, "q")
        await asyncio.sleep(0.1)   # filler fires after delay_secs=0.05
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
    s = make(False, thinking_secs=0.05)
    spoken = await collect(s, arm=lambda: s.on_llm_request(1, "what is quantum computing?"))
    assert len(spoken) == 1 and spoken[0] in CANNED_THINKING


@pytest.mark.asyncio
async def test_threshold_first_token_cancels_thinking():
    s = make(False, thinking_secs=5.0)
    spoken = await collect(s, 0.2, arm=lambda: (s.on_llm_request(1, "hello"),
                                                s.on_first_token(1)))
    assert spoken == []


@pytest.mark.asyncio
async def test_threshold_transcript_alone_does_not_speak():
    s = make(False, thinking_secs=0.05)
    spoken = await collect(s, arm=lambda: s.on_transcript(1, "hello"))
    assert spoken == []  # threshold mode: the LLM request is the trigger, not STT


@pytest.mark.asyncio
async def test_threshold_tool_filler_canned():
    s = make(False, tool_secs=0.05)
    spoken = await collect(s, 0.3, arm=lambda: (s.on_llm_request(1, "search news"),
                                                s.on_first_token(1),
                                                s.on_tool_running(1, "search_web")))
    assert len(spoken) == 1 and spoken[0] == TOOL_CANNED["search_web"]


# ---------------- shared behavior ----------------

@pytest.mark.asyncio
async def test_tool_done_cancels_pending_tool_filler():
    s = make(False, tool_secs=5.0)  # filler would fire long after the tool
    spoken = await collect(s, 0.3, arm=lambda: (s.on_llm_request(1, "q"),
                                                s.on_first_token(1),
                                                s.on_tool_running(1, "search_web"),
                                                s.on_tool_done(1)))
    assert spoken == []  # instant/finished tools never speak fillers


@pytest.mark.asyncio
async def test_disabled_never_speaks():
    s = make(True, enabled=False)
    spoken = await collect(s, arm=lambda: s.on_transcript(1, "hi"))
    assert spoken == []


def test_no_filler_model_anywhere():
    """The canned-only refactor must have removed the filler model entirely."""
    import inspect
    import agent.filler as f
    src = inspect.getsource(f)
    assert "OpenRouterClient" not in src
    assert "FILLER_MODEL" not in src
    assert "client.chat" not in src