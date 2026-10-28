"""Simple persistent notes (data/notes.json) — shared file, any process can use it."""
import json
import time
from pathlib import Path


class NotesStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> list:
        try:
            return json.loads(self.path.read_text(encoding="utf-8") or "[]")
        except (json.JSONDecodeError, OSError):
            return []

    def _write(self, items: list) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, text: str) -> dict:
        text = (text or "").strip()[:500]
        if not text:
            raise ValueError("note text is required")
        items = self._read()
        note = {"id": int(time.time() * 1000) % 1_000_000,
                "text": text, "ts": round(time.time(), 1)}
        items.append(note)
        self._write(items[-200:])   # cap history
        return note

    def read(self, limit: int = 10) -> str:
        items = self._read()[-limit:]
        if not items:
            return "(no notes)"
        return "\n".join(f"[{i['id']}] {i['text']}" for i in items)

    def clear(self) -> int:
        n = len(self._read())
        self._write([])
        return n