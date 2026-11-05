#!/usr/bin/env bash
# Raspberry Pi 5 audio setup: PipeWire echo cancellation (WebRTC AEC).
#
# Creates a virtual "Echo-Cancel Source" (mic) and "Echo-Cancel Sink" (speaker)
# so the OS subtracts the TTS output from the microphone BEFORE Pipecat sees it.
# This is the primary echo fix; the in-process echo detector (ECHO_DETECTOR) is
# only a fallback for machines without OS-level AEC (e.g. Windows dev PC).
#
# Run on the Pi:  bash scripts/setup_pi_audio.sh
set -euo pipefail

CONF=/etc/pipewire/pipewire.conf.d/99-echo-cancel.conf

if [ ! -f "$CONF" ]; then
  echo "==> installing PipeWire echo-cancel module config ($CONF)"
  sudo mkdir -p /etc/pipewire/pipewire.conf.d
  sudo tee "$CONF" >/dev/null <<'EOF'
# Virtual echo-cancelled source + sink (WebRTC AEC).
# Route TTS playback through the echo-cancel sink and read the mic from the
# echo-cancel source (device indices go in the project .env).
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
else
  echo "==> $CONF already present"
fi

echo "==> restarting PipeWire + WirePlumber"
systemctl --user restart pipewire wireplumber 2>/dev/null || systemctl --user restart pipewire
sleep 2

echo
echo "==> audio devices (pyaudio indices, same enumeration Pipecat uses):"
python3 "$(dirname "$0")/list_audio_devices.py" || true

echo
echo "==> also visible via pactl:"
pactl list sources short | sed 's/^/  source: /'
pactl list sinks short   | sed 's/^/  sink:   /'

echo
echo "==> next steps"
echo "1. In the project .env on the Pi set:"
echo "     AUDIO_INPUT_DEVICE=<index of 'Echo-Cancel Source'>"
echo "     AUDIO_OUTPUT_DEVICE=<index of 'Echo-Cancel Sink'>"
echo "     ECHO_DETECTOR=0                # OS AEC handles echo; detector not needed"
echo "2. Speak while the TTS plays — echo should be gone (give AEC a few seconds to converge after startup)."
echo "3. If the demo room still leaks a little echo, set ECHO_DETECTOR=1 as a second layer"
echo "   (costs up to ~400 ms of held mic audio after each bot response)."
