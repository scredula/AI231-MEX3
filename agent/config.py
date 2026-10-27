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
    TTS_PROVIDER = _env("TTS_PROVIDER", "deepgram")  # deepgram (uses ASR_API_KEY) | elevenlabs | openai
    TTS_API_KEY = _env("TTS_API_KEY")
    TTS_VOICE_ID = _env("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    TTS_MODEL = _env("TTS_MODEL", "aura-2-thalia-en")  # deepgram aura voice name; elevenlabs model id when TTS_PROVIDER=elevenlabs

    # GitHub
    GITHUB_TOKEN = _env("GITHUB_TOKEN")

    # Paths
    WORKSPACE_DIR = ROOT / _env("WORKSPACE_DIR", "workspace")
    TASKS_DB_PATH = ROOT / _env("TASKS_DB_PATH", "data/tasks.json")

    # Audio
    AUDIO_INPUT_DEVICE = _env("AUDIO_INPUT_DEVICE") or None
    AUDIO_OUTPUT_DEVICE = _env("AUDIO_OUTPUT_DEVICE") or None

    # Voice turn detection: seconds of silence before the agent decides you
    # finished speaking. Raise it if you pause mid-sentence and it keeps
    # cutting you off (lower = snappier, higher = more patient).
    TURN_STOP_SECS = float(_env("TURN_STOP_SECS", "0.8"))

    # Voice fillers: contextual "one moment…" lines spoken while the slow model
    # thinks or tools run. FILLER_MODEL should be a fast, non-thinking model.
    # FILLER_ALWAYS=1: the filler fires immediately after every transcript (the
    # real answer plays after it). 0 = legacy threshold behavior.
    FILLER_ENABLED = _env("FILLER_ENABLED", "1") == "1"
    FILLER_ALWAYS = _env("FILLER_ALWAYS", "1") == "1"
    THINKING_FILLER_SECS = float(_env("THINKING_FILLER_SECS", "3.5"))
    FILLER_AFTER_SECS = float(_env("FILLER_AFTER_SECS", "1.5"))
    FILLER_MAX_PER_TURN = int(_env("FILLER_MAX_PER_TURN", "3"))

    # Dashboard
    DASHBOARD_PORT = int(_env("DASHBOARD_PORT", "8900"))

    LOG_LEVEL = _env("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        cls.TASKS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
