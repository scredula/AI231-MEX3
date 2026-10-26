"""Task state model + structured events shared by the coordinator, MCP servers and voice agent."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import uuid


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    description: str
    state: TaskState = TaskState.QUEUED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = ""
    progress: str = ""
    result: str = ""
    events: list = field(default_factory=list)

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = self.created_at

    def add_event(self, kind: str, detail: str = "") -> dict:
        ev = {"time": datetime.now(timezone.utc).isoformat(), "kind": kind, "detail": detail}
        self.events.append(ev)
        self.updated_at = ev["time"]
        return ev

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["state"] = TaskState(d.get("state", TaskState.QUEUED.value))
        return cls(**d)


def new_task_id() -> str:
    return uuid.uuid4().hex[:10]


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
