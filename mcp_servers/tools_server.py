"""Custom MCP server: restricted filesystem, controlled command execution,
plus time/weather/system/reminders/notes/telegram tools.

All file paths are confined to the workspace (WORKSPACE_DIR). Commands are
checked against the allowlist/policy in coding_agent.policies. Designed so the
coding agent (or any MCP client) can work safely on the Raspberry Pi.

Run:  python -m mcp_servers.tools_server            (stdio)
      python -m mcp_servers.tools_server --http 8934
"""
import argparse
import asyncio
import base64
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from agent.config import Config
from coding_agent.policies import ExecutionPolicy, PolicyViolation

mcp = FastMCP("tools")
_policy = ExecutionPolicy(Config.WORKSPACE_DIR)
Config.ensure_dirs()


@mcp.tool()
def list_dir(path: str = ".") -> str:
    """List files/folders inside the workspace (relative path only)."""
    try:
        p = _policy.check_path(_policy.workspace / path)
        entries = sorted(x.name + ("/" if x.is_dir() else "") for x in Path(p).iterdir())
        return "\n".join(entries) or "(empty)"
    except PolicyViolation as e:
        raise ValueError(f"policy: {e}")


@mcp.tool()
def read_file(path: str) -> str:
    """Read a text file inside the workspace. Returns up to 20,000 characters."""
    try:
        p = _policy.check_path(_policy.workspace / path)
        return Path(p).read_text(encoding="utf-8")[:20000]
    except PolicyViolation as e:
        raise ValueError(f"policy: {e}")
    except FileNotFoundError:
        raise ValueError(f"not found: {path}")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write a text file inside the workspace (parents auto-created)."""
    try:
        p = _policy.check_path(_policy.workspace / path)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(content, encoding="utf-8")
        return f"ok: wrote {path} ({len(content)} chars)"
    except PolicyViolation as e:
        raise ValueError(f"policy: {e}")


@mcp.tool()
def run_command(command: str) -> str:
    """Run an allowlisted shell command in the workspace (timeout enforced)."""
    try:
        _policy.check_command(command)
    except PolicyViolation as e:
        raise ValueError(f"policy: {e}")
    try:
        proc = subprocess.run(command, shell=True, cwd=str(_policy.workspace),
                              capture_output=True, text=True, timeout=_policy.timeout)
        return f"exit={proc.returncode}\n{(proc.stdout + proc.stderr)[-2000:]}"
    except subprocess.TimeoutExpired:
        raise ValueError(f"command timed out after {_policy.timeout}s")



# ---- time / weather / system / reminders / notes / telegram ----

@mcp.tool()
def get_time(timezone: str = "") -> str:
    """Current date, time and weekday. Pass an IANA timezone (e.g. America/New_York)
    for that zone; omit for the machine's local time."""
    from services.timeutil import now_str
    return now_str(timezone)


@mcp.tool()
async def get_weather(location: str, units: str = "metric") -> str:
    """Current weather + today's forecast for a place name (Open-Meteo, no key).
    units: metric (°C, km/h) or imperial (°F, mph)."""
    from services.weather import get_weather as _gw, WeatherError
    try:
        return await _gw(location, units)
    except WeatherError as e:
        raise ValueError(str(e))


@mcp.tool()
def system_stats() -> str:
    """CPU load/temperature, memory, disk and uptime of this machine."""
    from services.sysstats import system_stats
    return system_stats()


@mcp.tool()
def add_reminder(text: str, in_minutes: float = 5.0) -> str:
    """Schedule a reminder. Fired by the running agent (spoken via TTS) and
    persisted in data/reminders.json. in_minutes: 0.1..10080 (7 days)."""
    from services.reminders import ReminderStore
    store = ReminderStore(Config.TASKS_DB_PATH.parent / "reminders.json")
    try:
        rem = store.add(text, in_minutes)
    except ValueError as e:
        raise ValueError(str(e))
    return f"reminder #{rem['id']} set: {rem['text']} (in {in_minutes} min)"


@mcp.tool()
def list_reminders() -> str:
    """List active (not yet fired) reminders."""
    from services.reminders import ReminderStore
    store = ReminderStore(Config.TASKS_DB_PATH.parent / "reminders.json")
    active = store.list_active()
    if not active:
        return "(no active reminders)"
    import time as _t
    return "\n".join(f"[{r['id']}] in {max(0, (r['due'] - _t.time()) / 60):.1f} min: {r['text']}"
                     for r in active)


@mcp.tool()
def cancel_reminder(reminder_id: int) -> str:
    """Cancel an active reminder by id (see list_reminders)."""
    from services.reminders import ReminderStore
    store = ReminderStore(Config.TASKS_DB_PATH.parent / "reminders.json")
    try:
        store.cancel(reminder_id)
    except ValueError as e:
        raise ValueError(str(e))
    return f"cancelled reminder {reminder_id}"


@mcp.tool()
def add_note(text: str) -> str:
    """Add a persistent note (data/notes.json). Survives restarts."""
    from services.notes import NotesStore
    try:
        note = NotesStore(Config.TASKS_DB_PATH.parent / "notes.json").add(text)
    except ValueError as e:
        raise ValueError(str(e))
    return f"note #{note['id']} saved"


@mcp.tool()
def read_notes(limit: int = 10) -> str:
    """Read the most recent notes (default 10)."""
    from services.notes import NotesStore
    return NotesStore(Config.TASKS_DB_PATH.parent / "notes.json").read(limit)


@mcp.tool()
async def send_telegram(text: str) -> str:
    """Send a message to the user's Telegram chat (TELEGRAM_BOT_TOKEN in .env)."""
    from services.telegram_client import send_message, TelegramError
    try:
        return await send_message(text)
    except TelegramError as e:
        raise ValueError(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP instead of stdio")
    args = parser.parse_args()
    if args.http:
        mcp.settings.port = args.http
        mcp.settings.host = "127.0.0.1"
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
