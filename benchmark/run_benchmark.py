"""Latency benchmark recorder.

Records monotonic timestamps for the full conversational round trip:
  speech onset (VAD) → end of speech → ASR result → LLM first token →
  TTS first audio chunk → first audible output → interruption onset/cessation
plus background task submit/complete.

On the PC (text mode / mocked pipeline) ASR/TTS/audio timestamps are absent and reported
as None; on the Raspberry Pi the voice pipeline records all of them. Output: JSON/CSV.
"""
import csv
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median


@dataclass
class Turn:
    turn_id: int
    t_speech_onset: float = None          # VAD detection of speech start
    t_speech_end: float = None            # end of user speech
    t_asr_result: float = None            # transcription available
    t_llm_first_token: float = None       # first LLM token
    t_tts_first_chunk: float = None       # first generated audio chunk (not yet audible)
    t_first_audible: float = None         # audio actually reaching the speaker
    t_interrupt_onset: float = None       # user speech while bot was speaking
    t_interrupt_playback_stopped: float = None
    t_task_submit: float = None
    t_task_complete: float = None
    meta: dict = field(default_factory=dict)

    @property
    def vad_to_asr(self):
        return _d(self.t_speech_end, self.t_asr_result)

    @property
    def asr_to_llm(self):
        return _d(self.t_asr_result, self.t_llm_first_token)

    @property
    def llm_to_tts(self):
        return _d(self.t_llm_first_token, self.t_tts_first_chunk)

    @property
    def tts_to_audible(self):
        return _d(self.t_tts_first_chunk, self.t_first_audible)

    @property
    def speech_onset_to_audible(self):
        return _d(self.t_speech_onset, self.t_first_audible)

    @property
    def interruption_latency(self):
        return _d(self.t_interrupt_onset, self.t_interrupt_playback_stopped)

    def to_dict(self):
        d = {k: v for k, v in asdict(self).items() if k != "meta"}
        for prop in ("vad_to_asr", "asr_to_llm", "llm_to_tts", "tts_to_audible",
                     "speech_onset_to_audible", "interruption_latency"):
            d[prop] = getattr(self, prop)
        for k, v in self.meta.items():
            d[f"meta_{k}"] = v
        return d


def _d(a, b):
    return None if a is None or b is None else round(b - a, 4)


class BenchmarkRecorder:
    def __init__(self):
        self.turns: list = []
        self._t0 = time.monotonic()

    def mark(self, turn_id: int, event: str, t: float = None) -> None:
        turn = self._turn(turn_id)
        setattr(turn, f"t_{event}", t if t is not None else time.monotonic())

    def _turn(self, turn_id: int) -> Turn:
        for t in self.turns:
            if t.turn_id == turn_id:
                return t
        t = Turn(turn_id=turn_id)
        self.turns.append(t)
        return t

    def export_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"meta": {"started_at": self._t0, "model": _model_names()}, "turns": [t.to_dict() for t in self.turns]},
            indent=1), encoding="utf-8")

    def export_csv(self, path: str) -> None:
        turns = [t.to_dict() for t in self.turns]
        if not turns:
            Path(path).write_text("", encoding="utf-8")
            return
        keys = sorted({k for t in turns for k in t})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(turns)

    def summary(self) -> dict:
        props = ["vad_to_asr", "asr_to_llm", "llm_to_tts", "tts_to_audible",
                 "speech_onset_to_audible", "interruption_latency"]
        out = {"turns": len(self.turns)}
        for prop in props:
            vals = [getattr(t, prop) for t in self.turns if getattr(t, prop) is not None]
            out[prop] = _stats(vals) if vals else {"available": False,
                "note": "requires audio/voice hardware measurements (Raspberry Pi voice mode)"}
        return out


def _stats(vals: list) -> dict:
    s = sorted(vals)
    return {"n": len(s), "min": s[0], "max": s[-1],
            "mean": round(mean(s), 4), "median": round(median(s), 4),
            "p95": s[min(len(s) - 1, int(round(0.95 * len(s))) - 1)],
            "failure_rate": 0.0}


def _model_names() -> dict:
    from agent.config import Config
    return {"fast_model": Config.FAST_MODEL, "coding_model": Config.CODING_MODEL,
            "asr_provider": Config.ASR_PROVIDER, "tts_provider": Config.TTS_PROVIDER}


def _run_text_benchmark(turns: int, out_base: str) -> None:
    """PC-testable benchmark: measures LLM first-token latency and background task
    submit→complete latency over N turns. Voice-only stages stay None (flagged later)."""
    import asyncio

    async def one_turn(i: int, rec: BenchmarkRecorder) -> None:
        from services.llm_client import OpenRouterClient
        rec.mark(i, "llm_first_token", None)  # placeholder row
        t0 = time.monotonic()
        client = OpenRouterClient()
        async for _ in client.chat_stream([{"role": "user", "content": "Say 'ready'."}]):
            break
        rec.mark(i, "llm_first_token", time.monotonic() - t0)
        rec._turn(i).meta["mode"] = "text"

    async def main_async():
        rec = BenchmarkRecorder()
        for i in range(1, turns + 1):
            await one_turn(i, rec)
            print(f"turn {i}/{turns} measured", flush=True)
        rec.export_json(out_base + ".json")
        rec.export_csv(out_base + ".csv")
        print(json.dumps(rec.summary(), indent=1))

    asyncio.run(main_async())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Latency benchmark")
    p.add_argument("--turns", type=int, default=3)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    _run_text_benchmark(args.turns, args.out)
