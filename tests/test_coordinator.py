import asyncio

import pytest

from agent.coordinator import Coordinator, TaskStore
from agent.task_events import Task, TaskState
from coding_agent.policies import ExecutionPolicy, PolicyViolation
from coding_agent.runner import PythonFallbackRunner
from benchmark.run_benchmark import BenchmarkRecorder, Turn
from tests.mocks import MockRunner

pytestmark = pytest.mark.asyncio


@pytest.fixture
def store(tmp_path):
    return TaskStore(tmp_path / "tasks.json")


@pytest.fixture
def coordinator(tmp_path):
    from coding_agent.runner import _BaseRunner  # ensure import path works
    made = {"fail": False}

    def factory():
        r = MockRunner(fail=made["fail"])
        made["runner"] = r
        return r

    c = Coordinator(TaskStore(tmp_path / "tasks.json"), factory)
    made["coordinator"] = c
    return made


async def test_submit_returns_immediately_and_completes(coordinator):
    c = coordinator["coordinator"]
    task_id = await c.submit("write a hello world")
    assert task_id
    t = await c.status(task_id)
    assert t.state in (TaskState.QUEUED, TaskState.RUNNING)
    # wait for completion
    for _ in range(50):
        t = await c.status(task_id)
        if t.state == TaskState.COMPLETED:
            break
        await asyncio.sleep(0.05)
    assert t.state == TaskState.COMPLETED
    assert t.result == "mock result"


async def test_state_transitions_record_events(coordinator):
    c = coordinator["coordinator"]
    tid = await c.submit("task with events")
    for _ in range(50):
        t = await c.status(tid)
        if t.state == TaskState.COMPLETED:
            break
        await asyncio.sleep(0.05)
    kinds = [e["kind"] for e in t.events]
    assert kinds[0] == "started"
    assert "progress" in kinds
    assert kinds[-1] == "completed"


async def test_failed_task_state(coordinator):
    c = coordinator["coordinator"]
    coordinator["fail"] = True
    tid = await c.submit("doomed task")
    for _ in range(50):
        t = await c.status(tid)
        if t.state == TaskState.FAILED:
            break
        await asyncio.sleep(0.05)
    assert t.state == TaskState.FAILED


async def test_cancel_pending_task_marks_cancelled(coordinator):
    c = coordinator["coordinator"]
    tid = await c.submit("to be cancelled")
    ok = await c.cancel(tid)
    assert ok
    t = await c.status(tid)
    assert t.state == TaskState.CANCELLED


async def test_cancel_unknown_task_returns_false(coordinator):
    c = coordinator["coordinator"]
    assert await c.cancel("nope") is False


async def test_store_roundtrip_and_persistence(store):
    task = Task(id="abc123", description="demo")
    task.add_event("started")
    store.upsert(task)
    loaded = store.get("abc123")
    assert loaded.description == "demo"
    assert loaded.events[0]["kind"] == "started"
    assert store.list()[0].id == "abc123"
    # second store instance sees the same data (MCP server use case)
    store2 = TaskStore(store.path)
    assert store2.get("abc123").description == "demo"


async def test_concurrent_submissions_keep_independent_state(coordinator):
    c = coordinator["coordinator"]
    ids = [await c.submit(f"task {i}") for i in range(4)]
    for _ in range(60):
        states = [(await c.status(i)).state for i in ids]
        if all(s == TaskState.COMPLETED for s in states):
            break
        await asyncio.sleep(0.05)
    results = [len((await c.status(i)).events) for i in ids]
    assert all(r >= 2 for r in results)
