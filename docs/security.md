# Security model and residual risks

## Controls implemented

- **Workspace confinement** — all file operations resolve inside `WORKSPACE_DIR`
  (`coding_agent/policies.ExecutionPolicy.check_path`); escapes are refused.
- **Command allowlist** — only allowlisted commands (`python`, `git`, `node`, `npm`, file
  utilities…) run in the workspace. Dangerous patterns (`sudo`, package-manager installs,
  `curl|sh`, raw-device writes, `rm -rf /`) are rejected.
- **Timeouts and bounded retries** — every command has a hard timeout; the fallback runner
  caps total steps (`MAX_RETRIES`); pi runs are bounded by an overall timeout and abortable.
- **Approval gates** — GitHub publishing requires the user's explicit approval in the
  conversation before `publish_website` runs; destructive operations are off the allowlist.
- **Cancellation** — cancel writes a flag file; event propagation stops for cancelled
  tasks and the pi process is aborted, so cancelled tasks cannot keep making changes.
- **Secret hygiene** — keys live only in `.env` (gitignored); they are never logged,
  never sent to the LLM in prompts, and never embedded in generated sites.
- **Execution log** — task events (`data/tasks.json`) record commands/progress without
  secret values.

## Residual risks (honest limits)

- This is a **policy layer, not a sandbox**. A determined LLM could still ask the
  allowlisted `python` to do anything Python can do inside the user's account; allowlists
  reduce accident likelihood, they do not contain a malicious payload. Strong isolation
  would require containers/user namespaces (Docker on a Pi adds overhead and its own
  risks; systemd `MemoryMax`/`CPUWeight` limits are provided as partial mitigation).
- `run_command` uses `shell=True` after allowlist/pattern checks — argument-level injection
  is possible for allowlisted binaries with tricky arguments.
- Generated code and external web content are never treated as trusted instructions, but
  prompt-injection via fetched content remains possible.
- Publishing is gated by conversation approval only; there is no cryptographic approval.

## If stronger isolation is needed later

Run `tools_server` and the coding agent under a dedicated Unix user with write access only
to the workspace, plus `systemd-run --scope -p MemoryMax=...`; or containerize with
read-only mounts. On the Pi, prefer the dedicated-user approach — Docker on ARM64 Pi is
workable but adds complexity for little gain here.
