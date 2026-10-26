"""Custom MCP server: background task status / cancellation.

Shares state with the in-process coordinator through the TaskStore JSON file
(TASKS_DB_PATH), so it works no matter which process hosts the coordinator.

Run:  python -m mcp_servers.task_server            (stdio)
      python -m mcp_servers.task_server --http 8933
"""
import argparse

from mcp.server.fastmcp import FastMCP

from agent.config import Config
from agent.coordinator import TaskStore

mcp = FastMCP("tasks")
_store = TaskStore(Config.TASKS_DB_PATH)


@mcp.tool()
def task_status(task_id: str) -> str:
    """Return state, latest progress and result of a background task."""
    t = _store.get(task_id)
    if not t:
        raise ValueError(f"unknown task id: {task_id}")
    return f"state={t.state.value} progress={t.progress!r} result={t.result[:200]!r}"


@mcp.tool()
def task_list() -> str:
    """List all background tasks and their states."""
    tasks = _store.list()
    if not tasks:
        return "no tasks yet"
    return "\n".join(f"{t.id}: {t.state.value} — {t.description[:60]}" for t in tasks)


@mcp.tool()
def task_cancel(task_id: str) -> str:
    """Request cancellation of a running background task."""
    if _store.request_cancel(task_id):
        return f"cancel requested for {task_id}"
    raise ValueError(f"task {task_id} cannot be cancelled (unknown or already finished)")


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
