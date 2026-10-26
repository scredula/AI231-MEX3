"""Direct workspace tools for the conversational agent: create/read/edit files, run
allowlisted commands. All paths confined to WORKSPACE_DIR; commands pass the execution
policy (allowlist + dangerous-pattern check + timeout). Shared by text and voice modes.
"""
import subprocess
from pathlib import Path

from coding_agent.policies import ExecutionPolicy, PolicyViolation
from agent.config import Config


class WorkspaceTools:
    def __init__(self, workspace: Path = None, timeout: int = 60):
        self.policy = ExecutionPolicy(workspace or Config.WORKSPACE_DIR, timeout=timeout)
        self.policy.workspace.mkdir(parents=True, exist_ok=True)

    def create_file(self, path: str, content: str) -> str:
        try:
            p = self.policy.check_path(path)
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text(content, encoding="utf-8")
            return f"ok: wrote {path} ({len(content)} chars)"
        except PolicyViolation as e:
            return f"error: policy — {e}"

    def read_file(self, path: str) -> str:
        try:
            p = self.policy.check_path(path)
            return Path(p).read_text(encoding="utf-8")[:10000]
        except PolicyViolation as e:
            return f"error: policy — {e}"
        except FileNotFoundError:
            return f"error: not found: {path}"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        try:
            p = self.policy.check_path(path)
            text = Path(p).read_text(encoding="utf-8")
        except PolicyViolation as e:
            return f"error: policy — {e}"
        except FileNotFoundError:
            return f"error: not found: {path}"
        if old_text not in text:
            return f"error: old_text not found in {path}"
        count = text.count(old_text)
        if count > 1:
            return f"error: old_text matches {count} times in {path} — include more context to make it unique"
        Path(p).write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"ok: edited {path}"

    def run_command(self, command: str) -> str:
        try:
            self.policy.check_command(command)
        except PolicyViolation as e:
            return f"error: policy — {e}"
        try:
            proc = subprocess.run(command, shell=True, cwd=str(self.policy.workspace),
                                  capture_output=True, text=True, timeout=self.policy.timeout)
            out = (proc.stdout + proc.stderr).strip()[-2000:]
            return f"exit={proc.returncode}\n{out}"
        except subprocess.TimeoutExpired:
            return f"error: timed out after {self.policy.timeout}s"

    def list_workspace(self) -> str:
        ws = self.policy.workspace
        entries = sorted(str(p.relative_to(ws)) + ("/" if p.is_dir() else "")
                         for p in ws.rglob("*") if ".git" not in p.parts)
        return "\n".join(entries[:200]) or "(empty)"