# AI231-MEX3 — Full-Duplex Voice AI Agent with a Background Coding Agent

A voice-first assistant built for a Raspberry Pi 5 (developed and tested on a PC):

- **Real-time conversational agent** (Pipecat): mic → Silero VAD → Deepgram ASR → OpenRouter LLM → ElevenLabs TTS → speaker, with barge-in interruption while the bot is speaking.
- **Background coding agent**: the [Pi coding agent](https://pi.dev) (`pi --mode rpc`) driven over a JSONL RPC protocol, configured through OpenRouter with a stronger model. A built-in Python fallback runner (OpenRouter tool-calling loop with restricted tools) is used automatically when `pi` is unavailable.
- **Task coordinator**: async task submission (instant task id), states `queued → running → completed/failed/cancelled`, progress announcements over TTS, status queries and cancellation — without blocking the audio pipeline.
- **Custom MCP servers**: ASR, TTS, task status/cancellation, and restricted filesystem/command execution (see `docs/mcp.md`).
- **Website build + GitHub publishing**: generate a site in the workspace, preview it, then publish to a private GitHub repo + Pages — only after explicit approval.
- **Latency benchmark** with CSV/JSON export (`benchmark/`).

## Quick start (PC, text mode — no audio hardware needed)

```bash
python3 -m venv .venv            # Python 3.11 or 3.12 recommended
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
cp .env.example .env             # fill in OPENROUTER_API_KEY (+ ASR/TTS keys for voice)
.venv/Scripts/python -m agent.main            # text-mode agent (Linux: .venv/bin/python)
```

Try: `build me a portfolio website`, `what's the status of task <id>`, `cancel task <id>`.

End-to-end non-interactive demo (real OpenRouter calls):
```bash
.venv/Scripts/python scripts/e2e_text_demo.py "Build a one-page portfolio website"
```

## Voice mode

```bash
.venv/Scripts/python -m agent.main --voice
```

Requires `ASR_API_KEY` (Deepgram) and `TTS_API_KEY` (ElevenLabs) plus a microphone/speaker.
Audio devices can be selected with `AUDIO_INPUT_DEVICE` / `AUDIO_OUTPUT_DEVICE` (PyAudio indices).

## Raspberry Pi 5 deployment (Raspberry Pi OS 64-bit)

```bash
sudo apt update && sudo apt install -y git
git clone <REPO_URL> AI231-MEX3 && cd AI231-MEX3
bash scripts/setup_pi.sh          # system deps, venv, pi CLI, .env template
nano .env                         # add keys, set audio device indices
bash scripts/start.sh             # run (TEXT=1 bash scripts/start.sh for text mode)
# later updates from the dev machine:
bash scripts/update_pi.sh
```

Optional boot service: `sudo cp deploy/voice-agent.service /etc/systemd/system/` (edit the
paths inside), then `sudo systemctl enable --now voice-agent`.

The coding agent runs as a Node.js subprocess (`pi`, arm64-compatible). If Node/pi is not
installed, the coordinator transparently uses the Python fallback runner — both honour
workspace confinement, command allowlists and cancellation.

## Latency benchmark

```bash
.venv/Scripts/python -m benchmark.run_benchmark --help      # record measurements
.venv/Scripts/python -m benchmark.analyze_results results.json
```

What is measurable where:

| Stage | PC (text/mock) | Pi (voice mode) |
|---|---|---|
| VAD speech onset/end | — | ✔ |
| ASR result latency | — | ✔ |
| LLM first token | ✔ (streamed client) | ✔ |
| TTS first chunk / first audible | — | ✔ |
| Interruption onset → playback stop | — | ✔ |
| Task submit → complete | ✔ | ✔ |

`analyze_results.py` reports min/mean/median/P95/max and flags stages that were not
sampled instead of inventing values.

## Repo layout

```
agent/          config, coordinator, text+voice agents, pipeline, interruption
coding_agent/   pi RPC adapter, fallback runner, execution policies
mcp_servers/    asr_server, tts_server, task_server, tools_server
services/       llm/asr/tts/github clients
benchmark/      run_benchmark, analyze_results
tests/          pytest suite (mock providers — no paid calls)
scripts/        setup_pi.sh, start.sh, stop.sh, update_pi.sh, e2e demo
deploy/         systemd unit
docs/           mcp.md, security.md, demo.md
```

## Security notes

See `docs/security.md`. In short: workspace confinement + command allowlist + timeouts +
approval-gated publishing. This is a policy layer, **not** a sandbox — residual risks are
documented there.

## What was tested vs. expected

- **Tested on PC**: full test suite (14 tests, mock providers), text-mode agent with real
  OpenRouter calls, background task submission/completion/cancellation, Pi RPC adapter
  creating real files, website build end-to-end, MCP servers importable and runnable.
- **Expected, not hardware-tested**: live microphone/speaker round trip, VAD/interruption
  on-device, Pi 5 memory/CPU behaviour. These need the physical Raspberry Pi; the code
  paths use Pipecat's documented 1.14 API and are unit-verified by import and mocked
  pipeline tests.


## Raspberry Pi audio: echo cancellation

Speaker setups hear the TTS through the microphone. The Pi fix is OS-level:

```bash
bash scripts/setup_pi_audio.sh     # installs PipeWire echo-cancel (WebRTC AEC)
```

Then point `.env` at the virtual devices it creates (see the script output,
or `python3 scripts/list_audio_devices.py`):

```
AUDIO_INPUT_DEVICE=<Echo-Cancel Source index>
AUDIO_OUTPUT_DEVICE=<Echo-Cancel Sink index>
ECHO_DETECTOR=0
```

The software echo detector (`agent/echo_detector.py`, `ECHO_DETECTOR=1`) is a
fallback for speaker setups without OS AEC — it holds mic audio during
echo-risk windows and cross-correlates it against the played TTS audio.


## Spotify playback (Premium account)

Voice control of Spotify: search + play, pause/resume, skip, volume, queue,
now-playing — via a custom MCP server (`mcp_servers/spotify_server.py`) and
direct voice tools, backed by the Spotify Web API (OAuth refresh flow).

**Native playback on the Pi**: `librespot` (built from source on the Pi, runs as
a systemd user service `librespot`) makes the Pi a Spotify Connect speaker named
**AI231-Pi**. Its output routes through the PipeWire echo-cancel sink, so the
AEC also cancels music from the microphone. Unit file: `deploy/librespot.service`.

Setup (one time):
1. Spotify app registration at developer.spotify.com/dashboard (redirect URI
   `http://127.0.0.1:8888/callback`), keys in `.env`
2. `python scripts/spotify_auth.py` — one-time browser login (token cache:
   `.spotify_cache`, copy to the Pi)
3. On the Pi: install librespot to `~/.local/bin`, copy
   `deploy/librespot.service` to `~/.config/systemd/user/`,
   `systemctl --user enable --now librespot`

Voice commands: *play <song/artist> [on the Pi]* · *pause the music* ·
*resume* · *next/previous track* · *volume 50* · *what's playing* ·
*play it on the Pi / on my phone*
