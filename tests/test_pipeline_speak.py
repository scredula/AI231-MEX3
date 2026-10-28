"""Regression guard: every TTSSpeakFrame in the voice pipeline MUST set
append_to_context=False. Default True leaks filler/announcement text into the
LLM context, and the model then narrates reasoning about its own fillers aloud
(the thought-leak bug)."""

from pathlib import Path

import agent.pipeline as pipeline_mod


def test_all_tts_speak_frames_exclude_context():
    src = Path(pipeline_mod.__file__).read_text(encoding="utf-8")
    sites = [line for line in src.splitlines() if "TTSSpeakFrame(" in line
             and "import" not in line and "isinstance" not in line]
    assert sites, "no TTSSpeakFrame construction sites found — check pipeline.py"
    for line in sites:
        assert "append_to_context=False" in line, (
            f"TTSSpeakFrame without append_to_context=False would leak into LLM context:\n{line}")


def test_system_prompt_says_think_silently():
    src = Path(pipeline_mod.__file__).read_text(encoding="utf-8")
    assert "Think silently" in src