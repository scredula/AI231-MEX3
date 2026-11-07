"""One-shot: reconstruct voice-side latencies from the dashboard event history
(no live collection needed). Run on the Pi; writes voice_latency.json.

NOTE: the bot's ANSWER frames carry a different turn id than the user's SPEECH
frames. pipecat serializes TTS, so the next bot_speech_start after a completed
speech turn belongs to that turn (FIFO commit)."""
import json
import statistics
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8900"


def main() -> None:
    if len(sys.argv) > 1:            # local dump mode: file with {"events": [...]}
        events = json.loads(open(sys.argv[1], encoding="utf-8").read())["events"]
    else:
        events = json.loads(urllib.request.urlopen(f"{BASE}/api/events", timeout=15).read())["events"]
    pending, turns = [], []
    done = set()
    for ev in events:
        t, turn = ev["type"], ev.get("turn")
        if t == "speech_start" and turn is not None and turn not in done:
            pending.append({"turn": turn, "st": {}})
            done.add(turn)
            continue
        if not pending:
            continue
        cur = pending[-1]["st"]      # events for the CURRENT speech turn
        if t == "speech_end" and "t_end" not in cur:
            cur["t_end"] = ev["t_ms"]
            if ev.get("speech_ms") is not None:
                cur["speech_ms"] = ev["speech_ms"]
        elif t == "transcript" and "t_transcript" not in cur:
            cur["t_transcript"] = ev["t_ms"]
            cur["text"] = ev.get("text", "")[:80]
        elif t == "bot_speech_start" and len(pending) > 0:
            while pending:               # may skip abandoned turns, then commit
                item = pending.pop(0)    # the next one with this same event
                st = item["st"]
                if "t_end" in st and "t_transcript" in st:
                    rec = {"turn": item["turn"],
                           "asr_final_ms": st["t_transcript"] - st["t_end"],
                           "e2e_ms": ev["t_ms"] - st["t_end"],
                           "speech_ms": st.get("speech_ms"),
                           "text": st.get("text", "")}
                    # ASR final measured from the estimated AUDIO end: the VAD's
                    # speech_end includes ~stop_secs of trailing silence, while
                    # Deepgram finals arrive eagerly (often before turn close).
                    audio_end = (st.get("t_start") or st["t_end"]) + (st.get("speech_ms") or 0)
                    rec["asr_final_ms"] = st["t_transcript"] - audio_end
                    rec["e2e_ms"] = ev["t_ms"] - st["t_end"]
                    if rec["e2e_ms"] >= 0:
                        turns.append(rec)
                        print(f"turn {rec['turn']}: asr {rec['asr_final_ms']}ms, "
                              f"e2e {rec['e2e_ms']}ms, speech {rec['speech_ms']}ms, "
                              f"\"{rec['text']}\"")
                    break

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