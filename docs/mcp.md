# MCP boundaries

Custom MCP servers (all run fine on the Pi — stdio transport, no ports needed):

| Server | Tools | Provider/config |
|---|---|---|
| `mcp_servers/asr_server.py` | `transcribe(wav_b64, sample_rate, channels, language)` | Deepgram or OpenAI via `ASR_PROVIDER`, `ASR_API_KEY` |
| `mcp_servers/tts_server.py` | `speak(text, voice_settings) -> base64 MP3` | ElevenLabs or OpenAI via `TTS_PROVIDER`, `TTS_API_KEY` |
| `mcp_servers/task_server.py` | `task_status`, `task_list`, `task_cancel` | shares `TASKS_DB_PATH` with the coordinator |
| `mcp_servers/tools_server.py` | `list_dir`, `read_file`, `write_file`, `run_command` | workspace-confined + allowlist (`coding_agent/policies.py`) |

Third-party servers can be added (e.g. filesystem, weather) — the task agent and
`tools_server` already cover the assignment's needs.

## ASR input contract (documented per assignment)

- Payload: base64-encoded WAV file, 16-bit PCM (`linear16`)
- Sample rates: 8000 / 16000 / 24000 / 48000 Hz
- Channels: 1 (mono) or 2 (stereo)
- Errors: invalid base64 / non-RIFF payload → tool error; provider failures/timeouts → tool error
- Streaming ASR: **not** implemented through MCP. See below.

## TTS output contract

- Input: `text` (1..5000 chars), optional `voice_settings` dict (provider-specific)
- Output: base64 MP3 (`audio/mpeg`)
- Streaming TTS: **not** implemented through MCP. See below.

## Why the live audio path does NOT go through MCP

MCP tool calls are request/response; per-chunk tool calls would add unacceptable latency
for streaming audio. The voice pipeline therefore uses Pipecat's **native streaming
integrations directly**:

- STT: `pipecat.services.deepgram.stt.DeepgramSTTService` (streaming WebSocket)
- TTS: `pipecat.services.elevenlabs.tts.ElevenLabsTTSService` (streaming WebSocket)
- LLM: `pipecat.services.openai.llm.OpenAILLMService` pointed at OpenRouter's
  OpenAI-compatible endpoint

MCP is used for one-shot capabilities (file transcription, on-demand synthesis, task
control, restricted execution) — the correct boundary per the assignment.
