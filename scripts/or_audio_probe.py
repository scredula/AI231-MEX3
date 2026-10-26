"""Probe OpenRouter audio endpoints with the models visible to this workspace."""
import base64
import io
import sys
import wave
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config

SPEECH = "https://openrouter.ai/api/v1/audio/speech"


def make_silence_wav() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    h = {"Authorization": f"Bearer {Config.OPENROUTER_API_KEY}"}
    # gpt-audio STT via chat/completions input_audio
    for model in ("openai/gpt-audio-mini", "openai/gpt-audio"):
        r = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=h, timeout=90,
                       json={"model": model, "max_tokens": 50,
                             "messages": [{"role": "user", "content": [
                                 {"type": "text", "text": "transcribe this audio"},
                                 {"type": "input_audio", "input_audio": {"data": make_silence_wav(), "format": "wav"}}]}]})
        print("STT", model, ":", r.status_code, r.text[:150])
    for model in ("openai/gpt-audio-mini", "openai/gpt-audio", "mistralai/voxtral-small-24b-2507"):
        r2 = httpx.post(SPEECH, headers=h, timeout=90,
                        json={"model": model, "input": "hello from openrouter", "response_format": "mp3"})
        print("TTS", model, ":", r2.status_code, r2.headers.get("content-type"), len(r2.content), "bytes",
              "" if r2.status_code == 200 else r2.text[:120])


if __name__ == "__main__":
    main()
