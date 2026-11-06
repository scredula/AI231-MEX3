"""Voice-side latency collector: run this while a human speaks phrases to the
agent. Polls the dashboard event history (localhost) and captures per-turn:

  asr_final_ms    end of speech -> transcript available (Deepgram final)
  e2e_ms          end of speech -> bot started speaking (answer audio)
  speech_ms       duration of the user's speech

Usage: .venv/bin/python benchmark/voice_bench_collect.py [seconds] [min_turns]
Output: benchmark/results/voice_latency.json
"""
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8900"
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 180
MIN_TURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def events() -> list:
    try:
        return json.loads(urllib.request.urlopen(f"{BASE}/api/events", timeout=5).read())["events"]
    except Exception:
        return []


def main() -> None:
    seen = 0
    turns = []
    state = {}
    deadline = time.monotonic() + SECONDS
    print(f"collecting voice turns for {SECONDS}s (speak naturally, one phrase at a time)...")
    while time.monotonic() < deadline and len(turns) < max(MIN_TURNS, 0) or (turns and len(turns) < MIN_TURNS and time.monotonic() < deadline + 60):
        evs = events()
        for i, ev in enumerate(evs):
            if i < seen:
                continue
            t = ev["type"]
            turn = ev.get("turn")
            if t == "speech_start" and turn not in state:
                state[turn] = {"t_start": ev["t_ms"]}
            elif t == "speech_end" and turn in state:
                state[turn]["t_end"] = ev["t_ms"]
                if ev.get("speech_ms") is not None:
                    state[turn]["speech_ms"] = ev["speech_ms"]
            elif t == "transcript" and turn in state:
                state[turn]["t_transcript"] = ev["t_ms"]
                state[turn]["text"] = ev.get("text", "")[:80]
            elif t == "bot_speech_start" and turn in state:
                st = state.pop(turn)
                if "t_end" in st and "t_transcript" in st:
                    rec = {"turn": turn,
                           "asr_final_ms": st["t_transcript"] - st["t_end"],
                           "e2e_ms": ev["t_ms"] - st["t_end"],
                           "speech_ms": st.get("speech_ms"),
                           "text": st.get("text", "")}
                    if all(v is not None and v >= 0 for v in (rec["asr_final_ms"], rec["e2e_ms"])):
                        turns.append(rec)
                        print(f"  captured: asr {rec['asr_final_ms']}ms, e2e {rec['e2e_ms']}ms, "
                              f"\"{rec['text']}\"", flush=True)
        seen = len(evs)
        time.sleep(0.4)

    def stats(vals):
        vals = sorted(vals)
        return {"n": len(vals), "mean": round(statistics.mean(vals)),
                "median": round(statistics.median(vals)),
                "p95": vals[min(len(vals) - 1, int(round(0.95 * len(vals))) - 1)],
                "min": vals[0], "max": vals[-1]}

    summary = {}
    if turns:
        summary = {
            "asr_final_ms": stats([t["asr_final_ms"] for t in turns]),
            "e2e_speech_to_audio_ms": stats([t["e2e_ms"] for t in turns]),
            "turns": turns,
            "note": "human-spoken phrases via microphone (VAD end -> transcript -> answer audio)",
        }
    out = Path(__file__).parent / "results" / "voice_latency.json"
    out.write_text(json.dumps({"summary": summary, "count": len(turns)}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "turns"}, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()