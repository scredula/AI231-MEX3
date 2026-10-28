"""Text-mode conversational agent: OpenRouter chat loop with coordinator + website tools.

Used for development/testing without audio hardware, and shares the tool layer with the
Pipecat voice pipeline (voice mode calls these same functions from frame handlers).
"""
import asyncio
import json
import logging
import re

from agent.config import Config
from agent.coordinator import Coordinator
from agent.task_events import TaskState
from agent.tools import WorkspaceTools
from services.llm_client import OpenRouterClient, LLMError
from services.websearch import search_web

log = logging.getLogger("conversation")

SYSTEM_PROMPT = (
    "You are a helpful voice-first assistant running locally on a Raspberry Pi 5. "
    "You can delegate build/code tasks to a background coding agent (submit_task) while staying "
    "available to chat. Keep spoken replies short (1-3 sentences). "
    "Available tools: submit_task, task_status, task_cancel, build_website, search_web, "
    "get_time, create_file, edit_file, read_file, list_workspace, run_command, publish_website.\n"
    "For small tasks (a quick script, a one-file fix) do them yourself with create_file/"
    "run_command; for bigger multi-step builds delegate with submit_task or build_website.\n"
    "Never read a tool's raw output aloud by itself — always work results into a natural "
    "sentence, and don't give a partial answer before you have everything you need.\n"
    "Publishing to GitHub always requires the user's explicit approval in the conversation — "
    "never call publish_website unless the user has clearly agreed to publish this exact site."
)

TOOLS = [
    {"type": "function", "function": {"name": "submit_task", "description": "Delegate a task to the background coding agent. Returns a task id immediately.",
     "parameters": {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}}},
    {"type": "function", "function": {"name": "task_status", "description": "Get status/progress of a background task.",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "task_cancel", "description": "Cancel a running background task.",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "string"}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "Get the current date, time and weekday. Pass an IANA timezone (e.g. America/New_York) to get that zone's time.",
     "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA timezone name, e.g. America/New_York. Omit for local time."}}, "required": []}}},
    {"type": "function", "function": {"name": "search_web", "description": "Search the web for current information (news, weather, prices, facts). Returns titles, snippets and URLs.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather and today's forecast for a location (Open-Meteo, live data).",
     "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City or place name"}, "units": {"type": "string", "enum": ["metric", "imperial"]}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "create_file", "description": "Create a file in the workspace (relative path only; parents auto-created).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact occurrence of old_text with new_text in a workspace file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file from the workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "list_workspace", "description": "List files in the workspace.",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run an allowlisted shell command (python, pip, git, node, file tools) in the workspace and get its output.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "build_website", "description": "Generate a small website in the workspace (background). Returns a task id.",
     "parameters": {"type": "object", "properties": {"request": {"type": "string"}}, "required": ["request"]}}},
    {"type": "function", "function": {"name": "publish_website", "description": "Publish a website from the workspace to GitHub. Requires explicit prior user approval.",
     "parameters": {"type": "object", "properties": {"project": {"type": "string"}, "repo_name": {"type": "string"}}, "required": ["project"]}}},
]


def _speakable(text: str, limit: int = 240) -> str:
    """Trim tool output into something TTS-friendly."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text[:limit]


class ConversationAgent:
    def __init__(self, coordinator: Coordinator, announcer=None):
        self.coordinator = coordinator
        self.client = OpenRouterClient()  # FAST_MODEL
        self.workspace_tools = WorkspaceTools()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.announcer = announcer or (lambda text: print(f"[agent] {text}"))

    def _tools(self) -> dict:
        return {"submit_task": self._submit_task,
                "task_status": self._task_status,
                "task_cancel": self._task_cancel,
                "build_website": self._build_website,
                "get_time": self._get_time,
                "search_web": self._search_web,
                "get_weather": self._get_weather,
                "create_file": self._create_file,
                "edit_file": self._edit_file,
                "read_file": self._read_file,
                "list_workspace": self._list_workspace,
                "run_command": self._run_command,
                "publish_website": self._publish_website}

    # tool implementations -------------------------------------------------
    async def _submit_task(self, description: str) -> str:
        task_id = await self.coordinator.submit(description)
        self.announcer(f"Okay, working on it. Task id {task_id}. I'll let you know when it's ready.")
        return json.dumps({"task_id": task_id, "state": "queued"})

    async def _task_status(self, task_id: str) -> str:
        t = await self.coordinator.status(task_id)
        if not t:
            return json.dumps({"error": f"unknown task {task_id}"})
        return json.dumps({"task_id": t.id, "state": t.state.value,
                           "progress": t.progress, "result": t.result})

    async def _task_cancel(self, task_id: str) -> str:
        ok = await self.coordinator.cancel(task_id)
        return json.dumps({"task_id": task_id, "cancelled": ok})

    async def _build_website(self, request: str) -> str:
        from agent.website import build_website_task
        task_id = await self.coordinator.submit(build_website_task(request))
        self.announcer("Building the website now — I'll tell you when it's ready to preview.")
        return json.dumps({"task_id": task_id})

    async def _get_time(self, timezone: str = "") -> str:
        from services.timeutil import now_str
        return now_str(timezone)

    async def _search_web(self, query: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, search_web, query)

    async def _get_weather(self, location: str, units: str = "metric") -> str:
        from services.weather import get_weather as _gw, WeatherError
        try:
            return await _gw(location, units)
        except WeatherError as e:
            return f"error: {e}"

    async def _create_file(self, path: str, content: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.workspace_tools.create_file, path, content)

    async def _edit_file(self, path: str, old_text: str, new_text: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.workspace_tools.edit_file, path, old_text, new_text)

    async def _read_file(self, path: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.workspace_tools.read_file, path)

    async def _list_workspace(self) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.workspace_tools.list_workspace)

    async def _run_command(self, command: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.workspace_tools.run_command, command)

    async def _publish_website(self, project: str, repo_name: str = "") -> str:
        from services.github_client import publish_website
        loop = asyncio.get_event_loop()
        try:
            urls = await loop.run_in_executor(None, publish_website, project, repo_name)
        except Exception as e:
            return json.dumps({"error": str(e)})
        self.announcer(f"Published. Repo: {urls['repo_url']} — site: {urls['pages_url']}")
        return json.dumps(urls)

    # main loop ------------------------------------------------------------
    async def reply(self, user_text: str) -> str:
        """Process one user turn, executing tools until a final answer. Returns reply text."""
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(6):
            try:
                msg = await self.client.chat(self.messages, tools=TOOLS)
            except LLMError as e:
                log.warning("LLM error: %s", e)
                self.messages.pop()
                return "Sorry, I couldn't reach the language service just now."
            self.messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or ""
            for call in calls:
                fn = self._tools().get(call["function"]["name"])
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = await fn(**args) if fn else json.dumps({"error": "unknown tool"})
                except TypeError as e:
                    result = json.dumps({"error": str(e)})
                self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                      "content": result if isinstance(result, str) else json.dumps(result)})
        return "I got stuck going back and forth — could you rephrase that?"
