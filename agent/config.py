"""Central configuration. Everything comes from environment / .env — nothing hardcoded."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, default).strip()
    return v


ROOT = Path(__file__).resolve().parent.parent


class Config:
    # LLM
    OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    FAST_MODEL = _env("FAST_MODEL", "google/gemini-2.5-flash")
    CODING_MODEL = _env("CODING_MODEL", "anthropic/claude-sonnet-4")

    # ASR
    ASR_PROVIDER = _env("ASR_PROVIDER", "deepgram")
    ASR_API_KEY = _env("ASR_API_KEY")
    ASR_MODEL = _env("ASR_MODEL", "nova-3")

    # TTS
    TTS_PROVIDER = _env("TTS_PROVIDER", "elevenlabs")
    TTS_API_KEY = _env("TTS_API_KEY")
    TTS_VOICE_ID = _env("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    TTS_MODEL = _env("TTS_MODEL", "eleven_turbo_v2_5")

    # GitHub
    GITHUB_TOKEN = _env("GITHUB_TOKEN")

    # Paths
    WORKSPACE_DIR = ROOT / _env("WORKSPACE_DIR", "workspace")
    TASKS_DB_PATH = ROOT / _env("TASKS_DB_PATH", "data/tasks.json")

    # Audio
    AUDIO_INPUT_DEVICE = _env("AUDIO_INPUT_DEVICE") or None
    AUDIO_OUTPUT_DEVICE = _env("AUDIO_OUTPUT_DEVICE") or None

    LOG_LEVEL = _env("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        cls.TASKS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
