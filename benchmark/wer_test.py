"""WER test set for the Machine Exercise evaluation table (§5).

Round-trip test: 10 reference phrases -> Deepgram TTS (same voice the agent
uses) -> Deepgram ASR (same model the agent uses) -> transcript -> WER vs the
reference. Measures ASR accuracy on clean synthetic speech; labeled as such in
the deck (a live human test set is future work).

Usage:  .venv/bin/python benchmark/wer_test.py
Output: benchmark/results/wer_results.json + printed table.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.config import Config  # noqa: E402

API = "https://api.deepgram.com/v1"
KEY = Config.ASR_API_KEY
HEADERS = {"Authorization": f"Token {KEY}", "Content-Type": "application/json"}

PHRASES = [
    "What time is it in New York?",
    "Tell me the weather in Manila tomorrow.",
    "Build me a small portfolio website.",
    "Search for the latest college football scores.",
    "Set a reminder for five minutes to check the oven.",
    "How many CPU cores does this machine have?",
    "Send a message to my Telegram saying hello.",
    "Cancel my background task please.",
    "The Gators host Oklahoma this Saturday at seven forty five PM.",
    "Read me the notes I saved earlier today.",
]

REFERENCES = [
    ["what", "time", "is", "it", "in", "new", "york"],
    ["tell", "me", "the", "weather", "in", "manila", "tomorrow"],
    ["build", "me", "a", "small", "portfolio", "website"],
    ["search", "for", "the", "latest", "college", "football", "scores"],
    ["set", "a", "reminder", "for", "five", "minutes", "to", "check", "the", "oven"],
    ["how", "many", "c", "p", "u", "cores", "does", "this", "machine", "have"],
    ["send", "a", "message", "to", "my", "telegram", "saying", "hello"],
    ["cancel", "my", "background", "task", "please"],
    ["the", "gators", "host", "oklahoma", "this", "saturday", "at", "seven", "forty", "five", "p", "m"],
    ["read", "me", "the", "notes", "i", "saved", "earlier", "today"],
]


def tts(text: str) -> bytes:
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{API}/speak?model={Config.TTS_MODEL}&encoding=linear16&container=wav&sample_rate=16000",
        data=body, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def asr(wav: bytes) -> str:
    model_q = f"model={Config.ASR_MODEL}&" if Config.ASR_MODEL else ""
    req = urllib.request.Request(
        f"{API}/listen?{model_q}punctuate=true&language=en",
        data=wav, headers={"Authorization": f"Token {KEY}",
                           "Content-Type": "audio/wav"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    return d["results"]["channels"][0]["alternatives"][0]["transcript"]


def wer(ref: list, hyp: list) -> tuple[int, int]:
    """Levenshtein word distance -> (errors, ref_len)."""
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, dp[j] = dp[j], cur
    return dp[n], m


def norm_words(s: str) -> list:
    repl = {"&": "and", "%": "percent", "$": "dollars"}
    for k, v in repl.items():
        s = s.replace(k, v)
    return [w.strip(".,!?;:") for w in s.lower().split()]


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    results = []
    total_err = total_len = 0
    for i, (phrase, ref) in enumerate(zip(PHRASES, REFERENCES), 1):
        t0 = time.monotonic()
        wav = tts(phrase)
        hyp_text = asr(wav)
        hyp = norm_words(hyp_text)
        # crude number normalization: map spoken digits that ASR may write as digits
        digit_map = {"2": "two", "3": "three", "4": "four", "5": "five", "7": "seven",
                     "8": "eight", "9": "nine", "10": "ten", "12": "twelve"}
        hyp = [digit_map.get(w, w) for w in hyp]
        err, n = wer(ref, hyp)
        total_err += err
        total_len += n
        results.append({"i": i, "reference": phrase, "transcript": hyp_text,
                        "wer": round(err / n * 100, 1), "errors": err, "words": n,
                        "tts_ms": int(wav and len(wav) / 32), "wall_ms": int((time.monotonic() - t0) * 1000)})
        print(f"[{i:2d}] WER {err}/{n} = {err / n * 100:5.1f}%  |  {hyp_text[:60]}")

    overall = round(total_err / total_len * 100, 1)
    print(f"\nOVERALL WER: {total_err}/{total_len} = {overall}%")
    (out_dir / "wer_results.json").write_text(json.dumps(
        {"overall_wer_pct": overall, "total_errors": total_err, "total_words": total_len,
         "asr_model": Config.ASR_MODEL, "tts_model": Config.TTS_MODEL,
         "note": "TTS->ASR round trip on synthetic speech (clean channel)",
         "runs": results}, indent=2), encoding="utf-8")
    print("saved:", out_dir / "wer_results.json")


if __name__ == "__main__":
    main()