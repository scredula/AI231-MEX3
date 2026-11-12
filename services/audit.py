"""Audit trail: durable, append-only JSONL log of everything the agent does.

One JSON line per event in data/audit/YYYY-MM-DD.jsonl:

  {"seq": 12, "session": "a1b2c3d4", "turn": 3, "type": "tool_result",
   "ts": "2026-11-12T18:10:22.123456+08:00", "mono": 1794446576.118, ...}

Design:
- Subscribes to the EventBus, so every dashboard event (transcripts, LLM
  requests, tool calls, latencies, token usage, errors) is audited with zero
  duplicate instrumentation. Anything publish()ed to the bus is audited.
- llm_request contexts are deduped: each message body is content-hashed and
  stored in full only the first time it appears ("c" field); later references
  carry only {"role", "h"}. Replaying the file reconstructs every request.
- Every line is secret-scrubbed (GitHub/PAT tokens, sk- keys, JWTs, Slack
  tokens, AWS keys, bearer headers, generic api_key/token/secret/password
  fields) BEFORE it touches disk.
- Daily files; files older than Config.AUDIT_RETAIN_DAYS (default 30) are
  deleted by a background prune.
- Writes happen on a background thread from a queue — the pipeline never
  blocks on disk.
"""
import hashlib
import json
import logging
import queue
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("audit")

# (pattern, replacement) applied to the serialized JSON line before writing.
_SECRET_PATTERNS = [
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"xox[bpars]-[A-Za-z0-9-]{10,}"), "[REDACTED_TOKEN]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)(authorization[\"']?\s*[:=]\s*[\"']?bearer\s+)[A-Za-z0-9._-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r'(?i)("(?:api[_-]?key|token|secret|password|authorization)"\s*:\s*")[^"]{6,}(")'), r'\1[REDACTED]\2'),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[A-Za-z0-9._~/+-]{12,}"), r"\1[REDACTED]"),
]

# first stores get truncated to this many characters (per message / per field)
_MAX_STORED = 8000


def scrub(text: str) -> str:
    for pat, repl in _SECRET_PATTERNS:
        text = pat.sub(repl, text)
    return text


class _ErrorNet(logging.Handler):
    """Catch ERROR+ log records from the service stack as audit events.

    pipecat / httpx / openai report API failures via logging, not frames —
    this is the net that turns them into auditable llm_error events.
    """
    def __init__(self, audit: "Audit"):
        super().__init__(level=logging.ERROR)
        self._audit = audit

    def emit(self, record: logging.LogRecord):
        try:
            self._audit.log({"type": "log_error", "logger": record.name,
                             "level": record.levelname,
                             "error": self.format(record)[:600]})
        except Exception:
            pass


def install_error_net(audit: "Audit") -> None:
    """Attach the audit net to the loggers that carry service failures.

    Never attaches to the audit logger itself — a failure while rendering an
    audit line must not re-enter the audit queue.
    """
    handler = _ErrorNet(audit)
    handler.setFormatter(logging.Formatter("%(message)s"))
    for name in ("pipecat", "openai", "httpx", "deepgram"):
        logging.getLogger(name).addHandler(handler)


class Audit:
    """EventBus sink -> JSONL files. Safe to create before any events exist."""

    def __init__(self, bus, audit_dir, retain_days: int = 30):
        self.dir = Path(audit_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retain_days = max(1, int(retain_days))
        self.session = uuid.uuid4().hex[:8]
        self._seq = 0
        self._turn = 0
        self._seen: dict[str, bool] = {}      # content hash -> True (per session)
        self._recent: deque = deque(maxlen=256)  # (type, turn, payload-hash) dup guard
        self._q = bus.subscribe()             # writer thread pumps this queue
        self._fh = None
        self._fh_date = None
        self._last_prune = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True, name="audit")
        self._thread.start()

    # ---- event entry ----------------------------------------------------
    def log(self, ev: dict) -> None:
        """Direct logging for events that don't flow through the EventBus."""
        self._q.put(dict(ev))

    # ---- writer ----------------------------------------------------------
    def _writer(self):
        prune_every = 6 * 3600
        self._last_prune = time.time()
        self._prune()
        while not self._stop.is_set() or not self._q.empty():
            try:
                ev = self._q.get(timeout=0.5)
            except queue.Empty:
                self._maybe_prune(prune_every)
                continue
            line = self._render(ev)
            if line:
                self._write(line)

    def _render(self, ev: dict):
        try:
            ev = dict(ev)
            # The pipecat observer sees each frame on more than one pipeline
            # traversal, so identical payloads can be published twice within
            # milliseconds (the dashboard masks this with seen-keys). The
            # audit must not record doubles: drop a repeat of the same
            # (type, turn, payload) within the recent window. Timing fields
            # differ between traversals, so they are excluded from the key;
            # llm_delta is never suppressed (two identical tokens in a row are
            # legitimate speech, not duplicates).
            skip = {"ts", "t_ms", "mono", "seq", "session", "duration", "latencies"}
            body = json.dumps({k: v for k, v in ev.items()
                               if k not in skip and not (isinstance(k, str) and k.endswith("_ms"))},
                              sort_keys=True, ensure_ascii=False, default=str)
            if ev.get("type") != "llm_delta":
                key = (ev.get("type"), ev.get("turn"),
                       hashlib.sha256(body[:2000].encode("utf-8")).hexdigest()[:16])
                if key in self._recent:
                    return None
                self._recent.append(key)
            self._seq += 1
            ev["seq"] = self._seq
            ev["session"] = self.session
            ev["ts"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")
            ev["mono"] = round(time.monotonic(), 3)
            if ev.get("type") == "turn_start":
                self._turn = ev.get("turn", self._turn)
            ev.setdefault("turn", self._turn)
            if ev.get("type") == "llm_request":
                ev["messages"] = self._dedupe_messages(ev.get("messages") or [])
            line = json.dumps(ev, ensure_ascii=False, default=str)
            return scrub(line)
        except Exception:
            log.exception("audit render failed")
            return None

    def _dedupe_messages(self, messages: list) -> list:
        out = []
        for m in messages:
            role = str(m.get("role", "?"))
            content = str(m.get("content", ""))[:_MAX_STORED]
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            entry = {"role": role, "h": h}
            if h not in self._seen:
                self._seen[h] = True
                entry["c"] = content
            out.append(entry)
        return out

    def _write(self, line: str):
        try:
            date = datetime.now().strftime("%Y-%m-%d")
            if self._fh is None or self._fh_date != date:
                if self._fh:
                    self._fh.close()
                self._fh = open(self.dir / f"{date}.jsonl", "a", encoding="utf-8")
                self._fh_date = date
            self._fh.write(line + "\n")
            self._fh.flush()
        except Exception:
            log.exception("audit write failed")

    # ---- rotation --------------------------------------------------------
    def _maybe_prune(self, every: float):
        if time.time() - self._last_prune >= every:
            self._last_prune = time.time()
            self._prune()

    def _prune(self):
        try:
            cutoff = datetime.now() - timedelta(days=self.retain_days)
            for f in self.dir.glob("*.jsonl"):
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    log.info("audit prune: deleted %s", f.name)
        except Exception:
            log.exception("audit prune failed")

    # ---- dashboard queries -------------------------------------------------
    def dates(self) -> list:
        out = []
        for f in sorted(self.dir.glob("*.jsonl")):
            try:
                out.append({"date": f.stem, "bytes": f.stat().st_size})
            except OSError:
                pass
        return out

    def query(self, date=None, type_=None, tool=None, turn=None, q=None,
              limit: int = 400, offset: int = 0) -> dict:
        """Filtered view of one audit file. Events come back oldest->newest.

        offset counts from the NEWEST matching event backwards (0 = latest),
        so page 1 always shows the most recent activity even as files grow.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        f = self.dir / f"{date}.jsonl"
        events: list = []
        types: dict = {}
        tools: dict = {}
        if f.exists():
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    t = ev.get("type", "?")
                    types[t] = types.get(t, 0) + 1
                    name = ev.get("name")
                    if not name and t == "tool_calls_started":
                        names = [c.get("name") for c in ev.get("calls", []) if c.get("name")]
                        name = names[0] if names else None
                        ev["_names"] = names
                    if name:
                        tools[name] = tools.get(name, 0) + 1
                    if type_ and t != type_:
                        continue
                    if tool and name != tool:
                        continue
                    if turn is not None and str(ev.get("turn")) != str(turn):
                        continue
                    if q and q.lower() not in line.lower():
                        continue
                    events.append(ev)
        total = len(events)
        if offset:
            events = events[:-offset] if offset < len(events) else []
        events = events[-limit:]
        return {"date": date, "total": total, "events": events,
                "types": types, "tools": tools}
