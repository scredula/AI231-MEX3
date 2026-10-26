"""Pipecat voice pipeline (full-duplex, interruption-capable).

Flow:  mic → VAD → STT(Deepgram) → LLM(OpenRouter FAST_MODEL) → TTS(ElevenLabs) → speaker

- Interruption: Pipecat raises an interruption whenever the user speaks while the bot is
  talking; user speech stays active during playback (LocalAudioTransport keeps the mic open).
- Background task announcements arrive through TTSSpeakFrame via the coordinator's
  announce sink — the audio pipeline is never blocked by task execution.
- The live audio path uses Pipecat's native streaming services directly; the custom MCP
  ASR/TTS servers are separate one-shot services (see docs/mcp.md).
"""
import asyncio
import logging

from agent.config import Config
from agent.observability import DashboardObserver, EventBus

log = logging.getLogger("pipeline")


def _llm_messages(context) -> list:
    """Serialize an LLMContext's messages for the dashboard (defensive, truncated)."""
    from agent.observability import _serialize_content

    out = []
    try:
        for m in context.get_messages():
            role = getattr(m, "role", None)
            if role is None and isinstance(m, dict):
                role = m.get("role", "?")
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            out.append({"role": str(role or "?"),
                        "content": _serialize_content(content)[:1200]})
    except Exception:
        logging.getLogger("observability").exception("llm message serialization failed")
        out.append({"role": "error", "content": "(failed to serialize messages)"})
    return out


class VoicePipelineAgent:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    async def run(self) -> None:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.frames.frames import TTSSpeakFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.worker import PipelineWorker
        from pipecat.workers.runner import WorkerRunner
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.services.deepgram.stt import DeepgramSTTService
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

        if not Config.OPENROUTER_API_KEY:
            raise SystemExit("OPENROUTER_API_KEY missing — copy .env.example to .env and fill it in")
        if not Config.ASR_API_KEY or not Config.TTS_API_KEY:
            raise SystemExit("ASR_API_KEY / TTS_API_KEY missing — the voice pipeline needs both")

        transport = LocalAudioTransport(LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            input_device_index=int(Config.AUDIO_INPUT_DEVICE) if Config.AUDIO_INPUT_DEVICE else None,
            output_device_index=int(Config.AUDIO_OUTPUT_DEVICE) if Config.AUDIO_OUTPUT_DEVICE else None,
        ))
        vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
        stt = DeepgramSTTService(api_key=Config.ASR_API_KEY, model=Config.ASR_MODEL)
        llm = OpenAILLMService(api_key=Config.OPENROUTER_API_KEY,
                               base_url=Config.OPENROUTER_BASE_URL,
                               settings=OpenAILLMService.Settings(model=Config.FAST_MODEL))
        tts = self._make_tts_service(transport_sample_rate=24000)

        # observability: dashboard + frame observer + LLM request capture
        from agent.observability import DashboardObserver, ObservableLLMService
        bus = EventBus()
        self.bus = bus
        observer = DashboardObserver(bus)
        llm._observer = observer
        llm.get_chat_completions = self._wrap_llm_get(llm)
        del ObservableLLMService  # documented mixin; hook wired manually above

        system = {
            "role": "system",
            "content": (
                "You are a friendly assistant running on a Raspberry Pi 5. Keep spoken answers "
                "short (1-3 sentences). You can delegate coding/build jobs to a background agent "
                "with submit_task and check on them with task_status. For the current time, use "
                "get_time; for weather use get_weather (live data); for other current events or facts, "
                "use search_web and summarize the "
                "results in one or two spoken sentences. For quick scripts or one-file fixes, "
                "use create_file, edit_file, read_file and run_command yourself; leave bigger "
                "multi-step builds to the background agent with submit_task. Never publish "
                "anything to GitHub unless the user explicitly approves in this conversation."
            ),
        }

        agent = self._make_tool_host()

        from pipecat.services.llm_service import FunctionCallParams

        async def submit_task(params: FunctionCallParams, description: str) -> None:
            """Delegate a task to the background coding agent. Returns the task id immediately.

            Args:
                description: What the background agent should build or do.
            """
            await params.result_callback(await agent.submit_task(description))

        async def task_status(params: FunctionCallParams, task_id: str) -> None:
            """Get the current state and latest progress of a background task.

            Args:
                task_id: The task id returned by submit_task.
            """
            await params.result_callback(await agent.task_status(task_id))

        async def task_cancel(params: FunctionCallParams, task_id: str) -> None:
            """Cancel a running background task.

            Args:
                task_id: The task id to cancel.
            """
            await params.result_callback(await agent.task_cancel(task_id))

        async def build_website(params: FunctionCallParams, request: str) -> None:
            """Build a small website in the workspace from a natural-language request.

            Args:
                request: What the website should contain.
            """
            await params.result_callback(await agent.build_website(request))

        async def get_time(params: FunctionCallParams) -> None:
            """Get the current local date and time, and the day of the week."""
            import datetime
            await params.result_callback(datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S"))

        async def search_web(params: FunctionCallParams, query: str) -> None:
            """Search the web for current information (news, prices, facts).

            Args:
                query: The search query.
            """
            import asyncio as _aio
            from services.websearch import search_web as _search
            result = await _aio.get_event_loop().run_in_executor(None, _search, query)
            await params.result_callback(result)

        async def get_weather(params: FunctionCallParams, location: str, units: str = "metric") -> None:
            """Get current weather and today's forecast for a location (live data).

            Args:
                location: City or place name.
                units: metric (Celsius) or imperial (Fahrenheit).
            """
            from services.weather import get_weather as _gw, WeatherError
            try:
                result = await _gw(location, units)
            except WeatherError as e:
                result = f"error: {e}"
            await params.result_callback(result)

        workspace_tools = self._make_workspace_tools()

        async def create_file(params: FunctionCallParams, path: str, content: str) -> None:
            """Create a file in the workspace (relative path only).

            Args:
                path: Workspace-relative file path.
                content: Full text content of the file.
            """
            import asyncio as _aio
            result = await _aio.get_event_loop().run_in_executor(
                None, workspace_tools.create_file, path, content)
            await params.result_callback(result)

        async def edit_file(params: FunctionCallParams, path: str, old_text: str, new_text: str) -> None:
            """Replace one exact occurrence of old_text with new_text in a workspace file.

            Args:
                path: Workspace-relative file path.
                old_text: Exact text to find (must be unique in the file).
                new_text: Replacement text.
            """
            import asyncio as _aio
            result = await _aio.get_event_loop().run_in_executor(
                None, workspace_tools.edit_file, path, old_text, new_text)
            await params.result_callback(result)

        async def read_file(params: FunctionCallParams, path: str) -> None:
            """Read a text file from the workspace.

            Args:
                path: Workspace-relative file path.
            """
            import asyncio as _aio
            result = await _aio.get_event_loop().run_in_executor(
                None, workspace_tools.read_file, path)
            await params.result_callback(result)

        async def run_command(params: FunctionCallParams, command: str) -> None:
            """Run an allowlisted shell command (python, pip, git, file tools) in the workspace.

            Args:
                command: The shell command to run.
            """
            import asyncio as _aio
            result = await _aio.get_event_loop().run_in_executor(
                None, workspace_tools.run_command, command)
            await params.result_callback(result)

        context = LLMContext([system], tools=[submit_task, task_status, task_cancel, build_website,
                                              get_time, search_web, get_weather, create_file,
                                              edit_file, read_file, run_command])
        aggregators = LLMContextAggregatorPair(context)

        pipeline = Pipeline([
            transport.input(),
            stt,
            vad,
            aggregators.user(),
            llm,
            tts,
            aggregators.assistant(),
            transport.output(),
        ])
        worker = PipelineWorker(pipeline, observers=[observer.pipecat_observer()])

        def announce(text: str) -> None:
            """Speak a task announcement without blocking the pipeline."""
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(worker.queue_frame(TTSSpeakFrame(text))))

        self.coordinator.on_announce(announce)

        from agent.dashboard import DashboardServer
        from pipecat.frames.frames import TranscriptionFrame

        async def say(text: str) -> None:
            """Inject typed text from the dashboard as a synthetic transcript.

            TranscriptionFrame (not TextFrame) so the aggregator treats it as user
            speech and TTS doesn't read it back aloud."""
            await worker.queue_frame(TranscriptionFrame(text, "", 16000))

        dashboard = DashboardServer(bus, say_handler=say)
        dashboard.start()
        log.info("conversation dashboard: %s", dashboard.url())

        runner = WorkerRunner()
        await runner.add_workers(worker)
        await runner.run()

    def _make_tool_host(self):
        """Tool implementations shared with the text-mode agent, minus its announcer."""
        from agent.conversation import ConversationAgent

        class _Host:
            def __init__(self, coordinator):
                self._inner = ConversationAgent(coordinator, announcer=lambda text: print(f"[announce] {text}"))

            async def submit_task(self, description: str) -> str:
                return await self._inner._submit_task(description)

            async def task_status(self, task_id: str) -> str:
                return await self._inner._task_status(task_id)

            async def task_cancel(self, task_id: str) -> str:
                return await self._inner._task_cancel(task_id)

            async def build_website(self, request: str) -> str:
                return await self._inner._build_website(request)

        return _Host(self.coordinator)

    def _make_tts_service(self, transport_sample_rate: int):
        """Build the streaming TTS service for the live audio path (provider from env)."""
        from pipecat.services.deepgram.tts import DeepgramTTSService
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        if Config.TTS_PROVIDER == "deepgram":
            if not Config.ASR_API_KEY:
                raise SystemExit("TTS_PROVIDER=deepgram needs ASR_API_KEY (one Deepgram account for ASR+TTS)")
            return DeepgramTTSService(
                api_key=Config.ASR_API_KEY,
                sample_rate=transport_sample_rate,
                settings=DeepgramTTSService.Settings(voice=Config.TTS_MODEL))
        if Config.TTS_PROVIDER == "elevenlabs":
            if not Config.TTS_API_KEY:
                raise SystemExit("TTS_PROVIDER=elevenlabs needs TTS_API_KEY (paid plan: library voices need a paid subscription)")
            return ElevenLabsTTSService(
                api_key=Config.TTS_API_KEY,
                settings=ElevenLabsTTSService.Settings(
                    voice=Config.TTS_VOICE_ID, model=Config.TTS_MODEL))
        raise SystemExit(f"unsupported TTS_PROVIDER {Config.TTS_PROVIDER!r} — use deepgram or elevenlabs")

    def _make_workspace_tools(self):
        from agent.tools import WorkspaceTools
        return WorkspaceTools()

    def _wrap_llm_get(self, llm):
        """Wrap llm.get_chat_completions to publish the exact LLM input for the dashboard."""
        import time as _time

        async def wrapped(context):
            obs = getattr(llm, "_observer", None)
            if obs is not None:
                obs.bus.publish({"type": "llm_request", "turn": obs.tracker.turn_id,
                                 "messages": _llm_messages(context)})
                obs.tracker.t_llm_request = _time.monotonic()
            return await llm.__class__.get_chat_completions(llm, context)

        return wrapped
