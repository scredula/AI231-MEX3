"""Analyze benchmark exports (JSON or CSV) produced by run_benchmark.py.

  python -m benchmark.analyze_results results.json
"""
import csv
import json
import sys
from statistics import mean, median
from pathlib import Path


def _stats(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {"n": len(s), "min": round(s[0], 4), "max": round(s[-1], 4),
            "mean": round(mean(s), 4), "median": round(median(s), 4),
            "p95": s[min(len(s) - 1, int(round(0.95 * len(s))) - 1)],
            "failure_rate": round(1 - len(s) / max(1, _n_total), 3) if (_n_total := None) else 0.0}


STAGES = ["vad_to_asr", "asr_to_llm", "llm_to_tts", "tts_to_audible",
          "speech_onset_to_audible", "interruption_latency"]


def load_rows(path: str) -> list:
    p = Path(path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))["turns"]
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(row, key):
    v = row.get(key)
    try:
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def analyze(rows: list) -> dict:
    out = {"turns": len(rows)}
    for stage in STAGES:
        vals = [v for row in rows if (v := _num(row, stage)) is not None]
        total = len(rows)
        if vals:
            s = sorted(vals)
            out[stage] = {"n": len(s), "min": round(s[0], 4), "max": round(s[-1], 4),
                          "mean": round(mean(s), 4), "median": round(median(s), 4),
                          "p95": s[min(len(s) - 1, int(round(0.95 * len(s))) - 1)],
                          "failure_rate": round((total - len(s)) / total, 3)}
        else:
            out[stage] = {"available": False,
                          "note": "no samples — this stage needs the voice pipeline (Pi hardware)"}
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result = analyze(load_rows(sys.argv[1]))
    print(json.dumps(result, indent=1))
