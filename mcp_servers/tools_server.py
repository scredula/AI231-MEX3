"""Custom MCP server: restricted filesystem + controlled command execution.

All paths are confined to the workspace (WORKSPACE_DIR). Commands are checked
against the allowlist/policy in coding_agent.policies. Designed so the coding
agent (or any MCP client) can work safely on the Raspberry Pi.

Run:  python -m mcp_servers.tools_server            (stdio)
      python -m mcp_servers.tools_server --http 8934
"""
import argparse
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
