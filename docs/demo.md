# Teacher demonstration script

Prerequisites: `.env` filled in (OPENROUTER_API_KEY required; ASR/TTS keys for the voice
demo), repo cloned on the Pi (or PC), test suite green.

## 1. Automated tests (~5 s)

```bash
.venv/Scripts/python -m pytest tests/ -q        # Linux: .venv/bin/python
```
Shows: coordinator state machine, cancellation, policy violations, benchmark export —
all with mock providers (no API cost).

## 2. Text agent + background coding agent (real LLM)

```bash
.venv/Scripts/python scripts/e2e_text_demo.py "Build me a one-page portfolio website for a student named Alex Ramirez"
```
Narrate: the conversational model answers and immediately delegates; a task id is
returned at once; the background coding agent (pi.dev, stronger model) creates
`workspace/site/index.html`, self-checks, and reports. Show the generated file and
`data/tasks.json` (task events: started → progress → completed).

## 3. Interactive text demo (optional)

```bash
.venv/Scripts/python -m agent.main
# you> build me a recipe website
# you> status <id from announce>
# you> cancel task <id>          (show cancellation mid-run)
```

## 4. MCP servers (one-shot tools)

```bash
# terminal A (HTTP transport so you can hit it from another shell):
.venv/Scripts/python -m mcp_servers.tools_server --http 8934
# terminal B: list the server's tools with any MCP client (e.g. `mcp` inspector or Hermes)
```
Demonstrate `run_command("python --version")` succeeding and
`run_command("sudo rm -rf /")` being rejected by policy.

## 5. Voice demo (needs mic + speaker)

```bash
.venv/Scripts/python -m agent.main --voice
```
Speak: *"build me a small website about houseplants"* → agent acknowledges aloud → while
it works, ask *"what's the weather like?"* (agent stays responsive) → when the task
finishes you hear the completion announcement → interrupt it mid-sentence by speaking
(bar-in). Show `logs/agent.log` for the announcement events.

## 6. Benchmark export

```bash
.venv/Scripts/python -m benchmark.analyze_results results.json
```
Explain which stages are measurable only on hardware (table in README).
