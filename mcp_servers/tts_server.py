"""Custom MCP server: speech synthesis.

Tool: speak
  input : text (1..5000 chars), optional voice_settings (provider-specific dict)
  output: MP3 audio, base64-encoded, audio/mpeg
Credentials via TTS_API_KEY env var.

Run:  python -m mcp_servers.tts_server            (stdio)
      python -m mcp_servers.tts_server --http 8932
"""
import argparse
import base64

from mcp.server.fastmcp import FastMCP

from services.tts_client import get_tts, TTSError

mcp = FastMCP("tts")


@mcp.tool()
async def speak(text: str, voice_settings: dict = None) -> str:
    """Synthesize speech. Args: text (plain text), voice_settings (optional dict,
    e.g. {"voice": "alloy"} for OpenAI or ElevenLabs voice settings). Returns
    base64-encoded MP3 audio (audio/mpeg)."""
    try:
        audio = await get_tts().speak(text, voice_settings)
    except TTSError as e:
        raise RuntimeError(str(e))
    return base64.b64encode(audio).decode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP instead of stdio")
    args = parser.parse_args()
    if args.http:
        mcp.settings.port = args.http
        mcp.settings.host = "127.0.0.1"
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
