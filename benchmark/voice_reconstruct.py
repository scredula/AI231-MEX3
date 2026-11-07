"""One-shot: reconstruct voice-side latencies from the dashboard event history
(no live collection needed). Run on the Pi; writes voice_latency.json."""
import json
import statistics
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8900"


def main() -> None:
    events = json.loads(urllib.request.urlopen(f"{BASE}/api/events", timeout=15).read())["events"]
    state, turns = {}, []
    for ev in events:
        t, turn = ev["type"], ev.get("turn")
        if turn is None:
            continue
        st = state.setdefault(turn, {})
        if t == "speech_start":
            st["t_start"] = ev["t_ms"]
        elif t == "speech_end":
            st["t_end"] = ev["t_ms"]
            if ev.get("speech_ms") is not None:
                st["speech_ms"] = ev["speech_ms"]
        elif t == "transcript":
            st["t_transcript"] = ev["t_ms"]
            st["text"] = ev.get("text", "")[:80]
        elif t == "bot_speech_start" and "t_end" in st and "t_transcript" in st:
            if "spoken" not in st:      # first answer only
                st["spoken"] = True
                rec = {"turn": turn,
                       "asr_final_ms": st["t_transcript"] - st["t_end"],
                       "e2e_ms": ev["t_ms"] - st["t_end"],
                       "speech_ms": st.get("speech_ms"),
                       "text": st.get("text", "")}
                if rec["asr_final_ms"] >= 0 and rec["e2e_ms"] >= 0:
                    turns.append(rec)
                    print(f"turn {turn}: asr {rec['asr_final_ms']}ms, e2e {rec['e2e_ms']}ms, "
                          f"speech {rec['speech_ms']}ms, \"{rec['text']}\"")

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
            "speech_durations_ms": stats([t["speech_ms"] for t in turns if t["speech_ms"]]),
            "note": "human-spoken phrases; VAD end -> transcript -> answer audio",
        }
    out = Path(__file__).parent / "results" / "voice_latency.json"
    out.write_text(json.dumps({"summary": summary, "turns": turns}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("saved:", out, f"({len(turns)} turns)")


if __name__ == "__main__":
    main()