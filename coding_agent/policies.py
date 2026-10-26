"""Execution policies: workspace confinement, command allowlist, timeouts, bounded retries.

Residual risk is documented in docs/security.md — this is policy-level restriction,
not a security sandbox. Never treat generated code or web content as trusted instructions.
"""
import re
import shlex
from pathlib import Path

DEFAULT_TIMEOUT = 120  # seconds per command
MAX_RETRIES = 3

# Commands the coding agent may run in the workspace (prefix match on first token)
DEFAULT_ALLOWLIST = [
    "python", "python3", "pip", "node", "npm", "npx",
    "git", "ls", "cat", "mkdir", "touch", "cp", "mv", "rm",
    "grep", "find", "head", "tail", "wc", "echo", "sed",
]

DANGEROUS_PATTERNS = [
    r"\bsudo\b", r"\bsu\b", r"\bchmod\b", r"\bchown\b",
    r"\bmkfs\b", r"\bdd\b", r"\bshutdown\b", r"\breboot\b",
    r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
    r"\brm\s+-rf\s+/(?!tmp|home)", r">\s*/dev/sd",
    r"\bapt\b", r"\bdpkg\b", r"\bpacman\b",
    r"curl\s", r"wget\s",
]


class PolicyViolation(Exception):
    pass


class ExecutionPolicy:
    def __init__(self, workspace: Path, allowlist: list = None, timeout: int = DEFAULT_TIMEOUT):
        self.workspace = Path(workspace).resolve()
        self.allowlist = allowlist or DEFAULT_ALLOWLIST
        self.timeout = timeout

    def check_command(self, command: str) -> None:
        """Raise PolicyViolation if the command is not allowed."""
        tokens = shlex.split(command)
        if not tokens:
            raise PolicyViolation("empty command")
        base = Path(tokens[0]).name
        if base not in self.allowlist:
            raise PolicyViolation(f"command '{base}' is not on the allowlist")
        low = command.lower()
        for pat in DANGEROUS_PATTERNS:
            if re.search(pat, low):
                raise PolicyViolation(f"command matches restricted pattern: {pat}")

    def check_path(self, path: Path) -> Path:
        """Resolve path and confine it inside the workspace."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        resolved = p.resolve()
        if not str(resolved).startswith(str(self.workspace)):
            raise PolicyViolation(f"path escapes workspace: {path}")
        return resolved
