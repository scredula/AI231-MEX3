#!/usr/bin/env bash
# Raspberry Pi 5 setup (Raspberry Pi OS 64-bit). Run once after cloning.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> System packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip portaudio19-dev alsa-utils git

echo "==> Python venv (Python 3.11+ required)"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> pi coding agent (Node.js)"
if ! command -v node >/dev/null 2>&1; then
  echo "Node.js not found. Install Node.js 20+ for arm64, e.g.:"
  echo "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
  echo "or: sudo apt-get install -y nodejs npm"
  exit 1
fi
npm install -g @mariozechner/pi-coding-agent || {
  echo "WARN: pi CLI install failed — the agent will fall back to the built-in Python runner."
}

echo "==> Configuration"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — edit it and add OPENROUTER_API_KEY, ASR_API_KEY, TTS_API_KEY."
fi

echo "==> Audio devices (for .env AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE)"
arecord -l 2>/dev/null || echo "no capture devices visible"
aplay -l 2>/dev/null || echo "no playback devices visible"

echo "==> Done. Next: edit .env, then run scripts/start.sh"
