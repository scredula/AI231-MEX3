"""Direct ASR provider clients (used by the ASR MCP server).

Accepts: 16-bit PCM WAV audio (mono preferred), sample rates 8k/16k/24k/48k.
Payload: raw bytes of the WAV file, sent as multipart/form-data (Deepgram) or
binary body (OpenAI). Streaming ASR is not implemented here — the live voice path
uses Pipecat's native streaming Deepgram integration instead (see docs/mcp.md).
"""
import logging
from typing import Optional

import httpx

from agent.config import Config

log = logging.getLogger("asr")

SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 48000)


class ASRError(Exception):
    pass


class DeepgramASR:
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.ASR_API_KEY
        self.model = model or Config.ASR_MODEL
        self.url = "https://api.deepgram.com/v1/listen"

    async def transcribe(self, wav_bytes: bytes, sample_rate: int = 16000,
                         channels: int = 1, language: str = "en") -> str:
        if not self.api_key:
            raise ASRError("ASR_API_KEY not configured")
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ASRError(f"unsupported sample rate {sample_rate}; use one of {SUPPORTED_SAMPLE_RATES}")
        params = {"model": self.model, "sample_rate": sample_rate,
                  "channels": channels, "language": language, "encoding": "linear16",
                  "container": "wav", "punctuate": "true"}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self.url, params=params,
                                  headers={"Authorization": f"Token {self.api_key}",
                                           "Content-Type": "audio/wav"},
                                  content=wav_bytes)
            if r.status_code != 200:
                raise ASRError(f"Deepgram error {r.status_code}: {r.text[:200]}")
            try:
                alt = r.json()["results"]["channels"][0]["alternatives"][0]
                return alt.get("transcript", "")
            except (KeyError, IndexError):
                return ""


class OpenAIASR:
    def __init__(self, api_key: str = None):
        from agent.config import Config
        self.api_key = api_key or Config.ASR_API_KEY
        self.url = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe(self, wav_bytes: bytes, sample_rate: int = 16000,
                         channels: int = 1, language: str = "en") -> str:
        if not self.api_key:
            raise ASRError("ASR_API_KEY not configured")
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self.url,
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                                  data={"model": "whisper-1", "language": language})
            if r.status_code != 200:
                raise ASRError(f"OpenAI error {r.status_code}: {r.text[:200]}")
            return r.json().get("text", "")


def get_asr():
    if Config.ASR_PROVIDER == "openai":
        return OpenAIASR()
    return DeepgramASR()
