"""Minimal OpenRouter chat-completions client (OpenAI-compatible) using httpx.

Used by the text-mode agent and the Python fallback task runner. Streams tokens.
Credentials come from env — never logged.
"""
import json
from typing import AsyncIterator

import httpx

from agent.config import Config


class LLMError(Exception):
    pass


class OpenRouterClient:
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        self.model = model or Config.FAST_MODEL
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.base_url = (base_url or Config.OPENROUTER_BASE_URL).rstrip("/")

    def _headers(self) -> dict:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY not configured")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(self, messages: list, tools: list = None, temperature: float = 0.7) -> dict:
        """Non-streaming call. Returns the assistant message dict (with optional tool_calls)."""
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{self.base_url}/chat/completions",
                                  headers=self._headers(), json=body)
            if r.status_code != 200:
                raise LLMError(f"OpenRouter error {r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]

    async def chat_stream(self, messages: list, temperature: float = 0.7) -> AsyncIterator[str]:
        """Streaming call yielding text deltas (for low first-token latency)."""
        body = {"model": self.model, "messages": messages, "temperature": temperature, "stream": True}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions",
                                     headers=self._headers(), json=body) as r:
                if r.status_code != 200:
                    text = (await r.aread()).decode(errors="replace")
                    raise LLMError(f"OpenRouter error {r.status_code}: {text[:200]}")
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
