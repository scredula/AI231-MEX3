"""Probe TTS providers directly: synthesizes a short clip to workspace/tts_probe.mp3."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from services.tts_client import get_tts


async def main() -> None:
    print("provider:", Config.TTS_PROVIDER, "| model:", Config.TTS_MODEL, "| voice:", Config.TTS_VOICE_ID)
    try:
        audio = await get_tts().speak("Hello, this is a text to speech test.")
        out = Config.WORKSPACE_DIR / "tts_probe.mp3"
        out.write_bytes(audio)
        print("OK:", len(audio), "bytes ->", out)
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e)[:300])


if __name__ == "__main__":
    asyncio.run(main())