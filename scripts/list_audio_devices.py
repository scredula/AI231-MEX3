#!/usr/bin/env python
"""List audio input/output devices with the indices Pipecat's LocalAudioTransport
(pyaudio) uses. Run on the Pi (or PC) to pick AUDIO_INPUT_DEVICE /
AUDIO_OUTPUT_DEVICE for .env — e.g. the PipeWire "Echo-Cancel Source"."""
import sys

try:
    import pyaudio
except ImportError:
    sys.exit("pyaudio not installed in this environment (pip install pyaudio)")

p = pyaudio.PyAudio()
try:
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        roles = []
        if (info.get("maxInputChannels") or 0) > 0:
            roles.append("IN ")
        if (info.get("maxOutputChannels") or 0) > 0:
            roles.append("OUT")
        if not roles:
            continue
        marks = ""
        try:
            if i == p.get_default_input_device_info()["index"]:
                marks += " *default-in*"
        except OSError:
            pass
        try:
            if i == p.get_default_output_device_info()["index"]:
                marks += " *default-out*"
        except OSError:
            pass
        print(f"[{i:2d}] {'/'.join(roles):7s} {info['name']}"
              f"  (in:{info.get('maxInputChannels', 0)} out:{info.get('maxOutputChannels', 0)})"
              f"{marks}")
finally:
    p.terminate()
