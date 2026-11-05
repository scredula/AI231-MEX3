#!/usr/bin/env bash
# Raspberry Pi 5 audio setup: PipeWire echo cancellation (WebRTC AEC).
#
# Creates a virtual "Echo-Cancel Source" (mic) and "Echo-Cancel Sink" (speaker)
# so the OS subtracts the TTS output from the microphone BEFORE Pipecat sees it,
# and makes them the system defaults. This is the primary echo fix; the
# in-process echo detector (ECHO_DETECTOR) is a second layer for residual echo.
#
# Run on the Pi:  bash scripts/setup_pi_audio.sh
# Idempotent: safe to run again (always rewrites the config).
set -euo pipefail
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

CONF=/etc/pipewire/pipewire.conf.d/99-echo-cancel.conf

echo "==> writing PipeWire echo-cancel module config ($CONF)"
sudo mkdir -p /etc/pipewire/pipewire.conf.d
sudo tee "$CONF" >/dev/null <<'EOF'
# Virtual echo-cancelled source + sink (WebRTC AEC).
# Route TTS playback through the echo-cancel sink and read the mic from the
# echo-cancel source (they are set as system defaults below).
context.modules = [
  {   name = libpipewire-module-echo-cancel
      args = {
        aec.args = { webrtc = { echo_suppression_level = 2 } }
        source.props = {
          node.name   = "echo_cancel_source"
          node.description = "Echo-Cancel Source"
        }
        sink.props = {
          node.name   = "echo_cancel_sink"
          node.description = "Echo-Cancel Sink"
        }
      }
  }
]
EOF

echo "==> restarting PipeWire + WirePlumber"
systemctl --user restart pipewire wireplumber
sleep 2

echo "==> setting echo-cancel pair as system defaults"
SRC_ID=$(wpctl status | grep -E '[0-9]+\. echo_cancel_source' | grep -oE '[0-9]+' | head -1)
SINK_ID=$(wpctl status | grep -E '[0-9]+\. echo_cancel_sink' | grep -oE '[0-9]+' | head -1)
if [ -n "${SRC_ID:-}" ] && [ -n "${SINK_ID:-}" ]; then
  wpctl set-default "$SRC_ID"
  wpctl set-default "$SINK_ID"
  echo "    defaults -> source #$SRC_ID (echo_cancel_source), sink #$SINK_ID (echo_cancel_sink)"
else
  echo "    WARNING: echo-cancel nodes not found — check: journalctl --user -u pipewire | tail -20" >&2
fi

echo
echo "==> audio devices (pyaudio indices, same enumeration Pipecat uses):"
python3 "$(dirname "$0")/list_audio_devices.py" 2>/dev/null || true

echo
echo "==> PipeWire graph check (expect a count > 0):"
wpctl status | grep -cE 'echo_cancel' || true

echo
echo "==> next steps"
echo "1. Keep AUDIO_INPUT_DEVICE / AUDIO_OUTPUT_DEVICE EMPTY in .env —"
echo "   the ALSA 'default' device routes through PipeWire's echo-cancel pair."
echo "2. ECHO_DETECTOR=1 is recommended as a second layer (residual echo)."
echo "3. Speak while the TTS plays — echo should be nearly gone."
