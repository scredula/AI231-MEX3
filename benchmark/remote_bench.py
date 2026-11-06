"""Remote benchmark: drives the RUNNING agent's dashboard with N phrases x M
runs via /say (text-injected turns) and collects per-stage latencies from
/api/events.

Measures (per turn):
  llm_ttft       transcript injection -> first LLM token
  llm_to_tts     first token          -> bot started speaking (first audio)
  total          injection            -> bot speech (spoken answer)

Voice-side stages (VAD->ASR etc.) require a real microphone turn; run
benchmark/voice_bench_collect.py alongside a human speaking to capture those.

Usage: .venv/bin/python benchmark/remote_bench.py [base_url] [runs]
Output: benchmark/results/latency_results.json
"""
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8900"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

PHRASES = [
    "What time is it in New York?",
    "What time is it in Manila?",
    "Tell me a one-sentence fun fact about octopuses.",
    "What is the capital of Australia?",
    "Name the largest planet in the solar system.",
    "What time is it in London?",
    "Say hello and introduce yourself briefly.",
    "What time is it in Berlin?",
]


def post_say(text: str) -> None:
    req = urllib.request.Request(
        f"{BASE}/say", method="POST",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)


def get_events() -> list:
    return json.loads(urllib.request.urlopen(f"{BASE}/api/events", timeout=10).read())["events"]


def collect_turn(events, start_idx, injected_text):
    """Find one complete turn after start_idx: llm_request -> first_token ->
    bot_speech_start. Returns dict of stage timings (ms) or None."""
    llm_req = ft = bot_speech = None
    for ev in events[start_idx:]:
        if ev["type"] == "llm_request" and llm_req is None:
            llm_req = ev["t_ms"]
        elif ev["type"] == "first_token" and llm_req is not None and ft is None:
            ft = ev["t_ms"]
        elif ev["type"] == "bot_speech_start" and ft is not None and bot_speech is None:
            bot_speech = ev["t_ms"]
            lat = ev.get("latencies") or {}
            break
    if not (llm_req and ft and bot_speech):
        return None
    return {"llm_ttft_ms": ft - llm_req,
            "llm_to_tts_ms": bot_speech - ft,
            "total_ms": bot_speech - llm_req}


def main() -> None:
    rows = []
    for run in range(1, RUNS + 1):
        for phrase in PHRASES:
            events = get_events()
            start_idx = len(events)
            post_say(phrase)
            turn = None
            for _ in range(60):          # up to 60 s for the turn to complete
                time.sleep(1)
                turn = collect_turn(get_events(), start_idx, phrase)
                if turn:
                    break
            if turn:
                turn.update({"run": run, "phrase": phrase})
                rows.append(turn)
                print(f"run {run}: {turn['llm_ttft_ms']}ms ttft, "
                      f"{turn['llm_to_tts_ms']}ms tts, {turn['total_ms']}ms total", flush=True)
            else:
                print(f"run {run}: NO TURN captured for {phrase!r}", flush=True)
            time.sleep(1)

    def stats(vals):
        vals = sorted(vals)
        return {"n": len(vals), "mean": round(statistics.mean(vals)),
                "median": round(statistics.median(vals)),
                "p95": vals[min(len(vals) - 1, int(round(0.95 * len(vals))) - 1)],
                "min": vals[0], "max": vals[-1]}

    summary = {
        "llm_ttft_ms": stats([r["llm_ttft_ms"] for r in rows]),
        "llm_to_tts_ms": stats([r["llm_to_tts_ms"] for r in rows]),
        "total_text_to_speech_ms": stats([r["total_ms"] for r in rows]),
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free (thinking disabled)",
        "note": "text-injected turns; voice-side stages measured separately",
        "captured": len(rows), "expected": RUNS * len(PHRASES),
    }
    out = Path(__file__).parent / "results" / "latency_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "turns": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print("saved:", out)


if __name__ == "__main__":
    main()