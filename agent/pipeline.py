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
from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame,
                                   InputAudioRawFrame)
from pipecat.processors.frame_processor import FrameProcessor


def _reasoning_body() -> dict:
    """OpenRouter reasoning params: exclude thinking from output, optional effort."""
    body: dict = {}
    if Config.REASONING_EXCLUDE:
        body["exclude"] = True
    if Config.REASONING_EFFORT:
        body["effort"] = Config.REASONING_EFFORT
    return body

log = logging.getLogger("pipeline")


class EchoGateProcessor(FrameProcessor):
    """Drops mic audio while the bot is speaking (+ a short tail) so the
    microphone can't pick up the speaker's TTS output and transcribe it back
    as user speech (the agent "hearing itself"). Trade-off: with the gate on,
    barge-in is impossible — speak after the bot finishes. Set
    MIC_GATE_DURING_TTS=0 (e.g. when using headphones) to restore barge-in.

    The bot-speech frames originate at the TTS service DOWNSTREAM of this
    processor, so state is fed in by the pipeline observer via
    on_bot_speech_start()/on_bot_speech_end().
    """

    def __init__(self, tail_secs: float = 0.6):
        super().__init__()
        self._tail = tail_secs
        self._muted = False
        self._unmute_task = None

    # ---- external state (called from the pipeline observer) ----
    def on_bot_speech_start(self) -> None:
        if self._unmute_task:
            self._unmute_task.cancel()
            self._unmute_task = None
        self._muted = True

    def on_bot_speech_end(self) -> None:
        async def unmute():
            await asyncio.sleep(self._tail)
            self._muted = False
        if self._unmute_task:
            self._unmute_task.cancel()
        self._unmute_task = asyncio.get_event_loop().create_task(unmute())

    # ---- frame processing ----
    async def process_frame(self, frame, direction) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and self._muted:
            return  # drop mic audio while the bot is speaking
        await self.push_frame(frame, direction)

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
        from pipecat.audio.vad.vad_analyzer import VADParams
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
        echo_gate = EchoGateProcessor(Config.MIC_GATE_TAIL_SECS) if Config.MIC_GATE_DURING_TTS else None
        from agent.echo_detector import (EchoDetectorProcessor, EchoState,
                                         EchoTranscriptGuard, TTSOutputTap)
        echo_state = EchoState()
        echo_detector = EchoDetectorProcessor(
            echo_state,
            corr_threshold=Config.ECHO_CORR_THRESHOLD,
            decision_window=Config.ECHO_DECISION_WINDOW,
            decide_ms=Config.ECHO_DECIDE_MS,
            silence_rms=Config.ECHO_SILENCE_RMS,
            silence_ms=Config.ECHO_SILENCE_MS,
            spec_threshold=Config.ECHO_SPEC_THRESHOLD) if Config.ECHO_DETECTOR else None
        echo_guard = EchoTranscriptGuard(
            echo_state,
            text_window=Config.ECHO_TEXT_WINDOW,
            text_similarity=Config.ECHO_TEXT_SIMILARITY) if Config.ECHO_DETECTOR else None
        tts_tap = TTSOutputTap(echo_state) if echo_detector else None
        vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(
            confidence=Config.VAD_CONFIDENCE,
            start_secs=Config.VAD_START_SECS,
            stop_secs=Config.TURN_STOP_SECS,
            min_volume=Config.VAD_MIN_VOLUME)))
        stt = DeepgramSTTService(api_key=Config.ASR_API_KEY, model=Config.ASR_MODEL)
        llm = OpenAILLMService(api_key=Config.OPENROUTER_API_KEY,
                               base_url=Config.OPENROUTER_BASE_URL,
                               settings=OpenAILLMService.Settings(
                                   model=Config.FAST_MODEL,
                                   extra={"extra_body": {"reasoning": _reasoning_body()}} if Config.REASONING_EXCLUDE or Config.REASONING_EFFORT else {}))
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
                "multi-step builds to the background agent with submit_task. Never read a "
                "tool's raw output aloud by itself — always work results into a natural "
                "sentence, and don't give a partial answer before you have everything you "
                "need. If a tool result is only intermediate context (like the current "
                "time), do not speak it — silently continue with your remaining tool "
                "calls and never comment on or correct what you said earlier. Think silently: "
                "never narrate, recap, or second-guess your reasoning out loud — your entire "
                "reply is spoken word-for-word, so output ONLY the final answer sentence(s), "
                "never 'Wait', 'Hmm', or deliberation about what to say. Never "
                "publish anything to GitHub unless the user explicitly approves "
                "in this conversation."
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

        async def get_time(params: FunctionCallParams, timezone: str = "") -> None:
            """Get the current date, time and weekday, optionally in an IANA timezone.

            Args:
                timezone: IANA timezone name, e.g. America/New_York. Omit for local time.
            """
            from services.timeutil import now_str
            await params.result_callback(now_str(timezone))

        async def set_reminder(params: FunctionCallParams, text: str, in_minutes: float = 5.0) -> None:
            """Set a reminder that will be spoken aloud when it fires.

            Args:
                text: What the reminder should say.
                in_minutes: Minutes from now (1 to 10080).
            """
            from services.reminders import ReminderStore
            store = ReminderStore(Config.TASKS_DB_PATH.parent / "reminders.json")
            try:
                rem = store.add(text, in_minutes)
            except ValueError as e:
                await params.result_callback(f"error: {e}")
                return
            await params.result_callback(f"Reminder set for {in_minutes} minutes from now.")

        async def list_reminders(params: FunctionCallParams) -> None:
            """List active reminders."""
            from services.reminders import ReminderStore
            store = ReminderStore(Config.TASKS_DB_PATH.parent / "reminders.json")
            import time as _t
            active = store.list_active()
            if not active:
                await params.result_callback("No active reminders.")
                return
            await params.result_callback("; ".join(
                f"{r['text']} in {max(0, (r['due'] - _t.time()) / 60):.0f} minutes" for r in active))

        async def add_note(params: FunctionCallParams, text: str) -> None:
            """Save a persistent note.

            Args:
                text: The note text.
            """
            from services.notes import NotesStore
            store = NotesStore(Config.TASKS_DB_PATH.parent / "notes.json")
            try:
                note = store.add(text)
            except ValueError as e:
                await params.result_callback(f"error: {e}")
                return
            await params.result_callback(f"Note saved.")

        async def read_notes(params: FunctionCallParams) -> None:
            """Read recent notes."""
            from services.notes import NotesStore
            await params.result_callback(
                NotesStore(Config.TASKS_DB_PATH.parent / "notes.json").read(5))

        async def system_stats(params: FunctionCallParams) -> None:
            """Get this machine's CPU, memory, disk and temperature stats."""
            from services.sysstats import system_stats as _stats
            await params.result_callback(await asyncio.get_event_loop().run_in_executor(None, _stats))

        async def send_telegram(params: FunctionCallParams, text: str) -> None:
            """Send a message to the user's Telegram.

            Args:
                text: The message to send.
            """
            from services.telegram_client import send_message, TelegramError
            try:
                result = await send_message(text)
            except TelegramError as e:
                await params.result_callback(f"couldn't send: {e}")
                return
            await params.result_callback(result)

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
                                              set_reminder, list_reminders, add_note, read_notes,
                                              system_stats, send_telegram,
                                              edit_file, read_file, run_command])
        aggregators = LLMContextAggregatorPair(context)

        pipeline = Pipeline([
            transport.input(),
            *( [echo_gate] if echo_gate else [] ),
            *( [echo_detector] if echo_detector else [] ),
            stt,
            vad,
            *( [echo_guard] if echo_guard else [] ),
            aggregators.user(),
            llm,
            tts,
            aggregators.assistant(),
            *( [tts_tap] if tts_tap else [] ),
            transport.output(),
        ])
        worker = PipelineWorker(pipeline, observers=[observer.pipecat_observer()])
        observer.echo_gate = echo_gate   # gate mutes mic while bot speaks
        observer.echo_detector = echo_detector  # feeds bot text for echo matching

        # Task announcements (append_to_context=False: spoken text must never enter
        # the LLM context).
        def announce(text: str) -> None:
            """Speak a task announcement without blocking the pipeline."""
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    worker.queue_frame(TTSSpeakFrame(text, append_to_context=False))))

        self.coordinator.on_announce(announce)

        # reminders: spoken through TTS when they fire
        from services.reminders import ReminderStore
        self._reminders = ReminderStore(
            Config.TASKS_DB_PATH.parent / "reminders.json",
            on_fire=lambda r: announce(f"Reminder: {r['text']}"))

        from agent.dashboard import DashboardServer
        from pipecat.frames.frames import TranscriptionFrame

        async def say(text: str) -> None:
            """Inject typed text from the dashboard as a synthetic transcript.

            TranscriptionFrame (not TextFrame) so the aggregator treats it as user
            speech and TTS doesn't read it back aloud. timestamp must be a str —
            an int crashes the RTVI turn-tracking observer's pydantic validation."""
            await worker.queue_frame(TranscriptionFrame(text, "", "0"))

        dashboard = DashboardServer(bus, say_handler=say)
        dashboard.start()
        log.info("conversation dashboard: %s", dashboard.url())

        runner = WorkerRunner()
        await runner.add_workers(worker)
        self._reminders.start()          # spoken reminders fire from here
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
                                 "t_ms": int(_time.time() * 1000),
                                 "model": Config.FAST_MODEL,
                                 "messages": _llm_messages(context)})
                obs.tracker.t_llm_request = _time.monotonic()
                obs.tracker.t_llm_first_token = None  # per-call stage timing
            return await llm.__class__.get_chat_completions(llm, context)

        return wrapped
