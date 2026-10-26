import json

import pytest

from agent.task_events import Task, TaskState
from coding_agent.runner import PythonFallbackRunner, build_prompt
from benchmark.run_benchmark import BenchmarkRecorder
from benchmark.analyze_results import analyze


def test_task_serialization_roundtrip():
    t = Task(id="x1", description="demo")
    t.add_event("progress", "half")
    t.state = TaskState.RUNNING
    d = json.loads(json.dumps(t.to_dict()))
    t2 = Task.from_dict(d)
    assert t2.state == TaskState.RUNNING
    assert t2.events[0]["detail"] == "half"


@pytest.mark.asyncio
async def test_fallback_runner_writes_file_and_runs_command(tmp_path):
    (tmp_path / "ws").mkdir()
    r = PythonFallbackRunner(tmp_path / "ws")
    events = []

    async def fake_chat(messages, tools=None, temperature=0.7):
        calls = [m for m in messages if m.get("role") == "assistant"]
        if not calls:
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "write_file",
                 "arguments": json.dumps({"path": "site/index.html", "content": "<h1>hi</h1>"})}}]}
        return {"role": "assistant", "content": "built it"}

    r.client.chat = fake_chat
    task = Task(id="t1", description="make a site")
    result = await r.run(task, lambda k, d="": events.append((k, d)))
    assert result == "built it"
    assert (tmp_path / "ws" / "site" / "index.html").read_text() == "<h1>hi</h1>"
    assert any(k == "progress" for k, _ in events)


@pytest.mark.asyncio
async def test_fallback_runner_policy_violation_reported_to_llm(tmp_path):
    (tmp_path / "ws").mkdir()
    r = PythonFallbackRunner(tmp_path / "ws")

    async def fake_chat(messages, tools=None, temperature=0.7):
        calls = [m for m in messages if m.get("role") == "assistant"]
        if not calls:
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "run_command",
                 "arguments": json.dumps({"command": "sudo rm -rf /"})}}]}
        # the tool result fed back should mention the policy violation
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert "policy violation" in tool_msgs[-1]["content"]
        return {"role": "assistant", "content": "gave up, no unauthorized changes"}

    r.client.chat = fake_chat
    result = await r.run(Task(id="t2", description="break out"), lambda *a: None)
    assert "no unauthorized changes" in result


def test_benchmark_recorder_and_analyzer(tmp_path):
    rec = BenchmarkRecorder()
    rec.mark(1, "speech_onset", 100.0)
    rec.mark(1, "asr_result", 100.5)
    rec.mark(1, "llm_first_token", 100.9)
    rec.mark(2, "speech_onset", 200.0)
    rec.mark(2, "asr_result", 200.8)
    rec.mark(2, "llm_first_token", 201.2)
    j = tmp_path / "res.json"
    c = tmp_path / "res.csv"
    rec.export_json(str(j))
    rec.export_csv(str(c))
    rows = json.loads(j.read_text())["turns"]
    assert len(rows) == 2
    summary = analyze(rows)
    assert summary["turns"] == 2
    assert summary["asr_to_llm"]["median"] > 0
    # TTS stages absent (text-mode benchmark) must be flagged, not fabricated
    assert summary["tts_to_audible"]["available"] is False
    # CSV path parses the same
    from benchmark.analyze_results import load_rows
    assert analyze(load_rows(str(c)))["turns"] == 2


def test_build_prompt_mentions_workspace():
    from agent.task_events import Task
    p = build_prompt(Task(id="x", description="portfolio"))
    assert "portfolio" in p
