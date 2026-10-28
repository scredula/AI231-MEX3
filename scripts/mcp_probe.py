"""MCP stdio probe: launches task_server and tools_server as real subprocesses,
performs the MCP handshake and calls tools. Verifies the servers actually work."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def probe(server_module: str, calls: list) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", server_module],
                                   cwd=str(Path(__file__).resolve().parent.parent))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            print(f"[{server_module}] tools:", [t.name for t in tools.tools])
            for name, args in calls:
                try:
                    res = await s.call_tool(name, args)
                    texts = "".join(c.text for c in res.content if hasattr(c, "text"))
                    print(f"[{server_module}] {name}({args}) -> {texts[:120]}")
                except Exception as e:
                    print(f"[{server_module}] {name} -> ERROR {type(e).__name__}: {str(e)[:120]}")


async def main() -> None:
    await probe("mcp_servers.tools_server", [
        ("write_file", {"path": "mcp_probe.txt", "content": "hello from mcp"}),
        ("read_file", {"path": "mcp_probe.txt"}),
        ("list_dir", {"path": "."}),
        ("run_command", {"command": "python --version"}),
        ("run_command", {"command": "sudo rm -rf /"}),  # must be rejected
        ("get_time", {"timezone": "America/New_York"}),
        ("get_weather", {"location": "New York"}),
        ("system_stats", {}),
        ("add_note", {"text": "mcp probe note"}),
        ("read_notes", {}),
        ("add_reminder", {"text": "mcp probe reminder", "in_minutes": 2}),
        ("list_reminders", {}),
        ("send_telegram", {"text": "voice agent mcp probe"}),  # clean error until token set
    ])
    await probe("mcp_servers.task_server", [
        ("task_list", {}),
    ])


asyncio.run(main())
