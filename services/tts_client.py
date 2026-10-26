"""Direct TTS provider clients (used by the TTS MCP server).

Input: plain text + optional voice settings.
Output: audio/mpeg (MP3) bytes. Streaming TTS is on the live voice path via
Pipecat's native ElevenLabs/OpenAI TTS services — not through MCP (see docs/mcp.md).
"""
import logging

import httpx

from agent.config import Config

log = logging.getLogger("tts")


class TTSError(Exception):
    pass


class ElevenLabsTTS:
    def __init__(self, api_key: str = None, voice_id: str = None, model: str = None):
        self.api_key = api_key or Config.TTS_API_KEY
        self.voice_id = voice_id or Config.TTS_VOICE_ID
        self.model = model or Config.TTS_MODEL

    async def speak(self, text: str, voice_settings: dict = None) -> bytes:
        if not self.api_key:
            raise TTSError("TTS_API_KEY not configured")
        if not text or len(text) > 5000:
            raise TTSError("text must be 1..5000 characters")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        body = {"text": text, "model_id": self.model,
                "voice_settings": voice_settings or {"stability": 0.5, "similarity_boost": 0.75}}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, headers={"xi-api-key": self.api_key,
                                                "Content-Type": "application/json"},
                                  json=body)
            if r.status_code != 200:
                raise TTSError(f"ElevenLabs error {r.status_code}: {r.text[:200]}")
            return r.content


class OpenAITTS:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.TTS_API_KEY

    async def speak(self, text: str, voice_settings: dict = None) -> bytes:
        if not self.api_key:
            raise TTSError("TTS_API_KEY not configured")
        if not text or len(text) > 4096:
            raise TTSError("text must be 1..4096 characters")
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post("https://api.openai.com/v1/audio/speech",
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  json={"model": "gpt-4o-mini-tts", "input": text,
                                        "voice": (voice_settings or {}).get("voice", "alloy"),
                                        "response_format": "mp3"})
            if r.status_code != 200:
                raise TTSError(f"OpenAI error {r.status_code}: {r.text[:200]}")
            return r.content


def get_tts():
    if Config.TTS_PROVIDER == "openai":
        return OpenAITTS()
    return ElevenLabsTTS()
