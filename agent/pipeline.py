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

log = logging.getLogger("pipeline")


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
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
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
        tts = ElevenLabsTTSService(api_key=Config.TTS_API_KEY,
                                   settings=ElevenLabsTTSService.Settings(
                                       voice=Config.TTS_VOICE_ID, model=Config.TTS_MODEL))

        system = {
            "role": "system",
            "content": (
                "You are a friendly assistant running on a Raspberry Pi 5. Keep spoken answers "
                "short (1-3 sentences). You can delegate coding/build jobs to a background agent "
                "with submit_task and check on them with task_status. Never publish anything to "
                "GitHub unless the user explicitly approves in this conversation."
            ),
        }

        agent = self._make_tool_host()

        from pipecat.services.llm_service import FunctionCallParams

        async def submit_task(params: FunctionCallParams, description: str) -> str:
            """Delegate a task to the background coding agent. Returns the task id immediately.

            Args:
                description: What the background agent should build or do.
            """
            return await agent.submit_task(description)

        async def task_status(params: FunctionCallParams, task_id: str) -> str:
            """Get the current state and latest progress of a background task.

            Args:
                task_id: The task id returned by submit_task.
            """
            return await agent.task_status(task_id)

        async def task_cancel(params: FunctionCallParams, task_id: str) -> str:
            """Cancel a running background task.

            Args:
                task_id: The task id to cancel.
            """
            return await agent.task_cancel(task_id)

        async def build_website(params: FunctionCallParams, request: str) -> str:
            """Build a small website in the workspace from a natural-language request.

            Args:
                request: What the website should contain.
            """
            return await agent.build_website(request)

        context = LLMContext([system], tools=[submit_task, task_status, task_cancel, build_website])
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
        worker = PipelineWorker(pipeline)

        def announce(text: str) -> None:
            """Speak a task announcement without blocking the pipeline."""
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(worker.queue_frame(TTSSpeakFrame(text))))

        self.coordinator.on_announce(announce)

        runner = WorkerRunner()
        await runner.add_workers([worker])
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
