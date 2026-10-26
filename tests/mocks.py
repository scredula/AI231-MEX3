"""Mock LLM / ASR / TTS providers so tests run without paid API calls."""
from services.llm_client import LLMError  # noqa: F401  (kept import surface stable)


class MockLLMClient:
    """Scripted tool-calling loop client standing in for OpenRouterClient."""

    def __init__(self, script: list):
        self.script = script  # list of message dicts returned in order
        self.calls: list = []

    async def chat(self, messages, tools=None, temperature=0.7):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            return {"role": "assistant", "content": "done"}
        return self.script.pop(0)

    async def chat_stream(self, messages, temperature=0.7):
        yield "ok"


class MockRunner:
    """Fake background runner: records on_event calls, honors cancellation flags."""

    name = "mock"

    def __init__(self, fail: bool = False):
        self.fail = fail

    async def run(self, task, on_event):
        import asyncio, os
        on_event("progress", "working")
        await asyncio.sleep(0.05)
        if os.path.exists(task.id + ".cancel"):
            raise asyncio.CancelledError()
        if self.fail:
            raise RuntimeError("mock failure")
        return "mock result"

    async def abort(self):
        pass


def make_wav(seconds: float = 0.2, rate: int = 16000) -> bytes:
    """Minimal valid mono 16-bit PCM WAV payload (silence) for ASR tests."""
    import struct, io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()
