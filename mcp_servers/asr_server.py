"""Custom MCP server: speech recognition.

Tool: transcribe
  input : wav_bytes (base64 WAV file; 16-bit PCM, mono recommended,
          sample_rate in {8000,16000,24000,48000})
  output: transcribed text
Provider configured via ASR_PROVIDER / ASR_API_KEY env vars. Errors are returned
as structured tool errors (invalid input, provider failure, timeout).

Run:  python -m mcp_servers.asr_server            (stdio transport)
      python -m mcp_servers.asr_server --http 8931 (streamable HTTP on a port)
"""
import argparse
import base64
import binascii

from mcp.server.fastmcp import FastMCP

from services.asr_client import get_asr, ASRError

mcp = FastMCP("asr")


@mcp.tool()
async def transcribe(wav_b64: str, sample_rate: int = 16000, channels: int = 1,
                     language: str = "en") -> str:
    """Transcribe a WAV file (16-bit PCM, base64-encoded) to text.

    Accepts mono or stereo, sample rates 8000/16000/24000/48000 Hz. Returns the
    transcript, or raises a tool error on invalid audio / provider failure."""
    try:
        wav_bytes = base64.b64decode(wav_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("wav_b64 is not valid base64")
    if not wav_bytes.startswith(b"RIFF"):
        raise ValueError("expected a RIFF/WAV payload")
    try:
        return await get_asr().transcribe(wav_bytes, sample_rate, channels, language)
    except ASRError as e:
        raise RuntimeError(str(e))


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
