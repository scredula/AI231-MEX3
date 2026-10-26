"""Constructs the full voice pipeline (transport, VAD, STT, LLM, TTS, context+tools)
without starting the audio device — verifies registration of everything that the
runtime needs, including Pipecat direct-function tool schemas."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Config
from agent.main import _make_coordinator


async def main() -> None:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.services.llm_service import FunctionCallParams
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

    transport = LocalAudioTransport(LocalAudioTransportParams())  # in/out disabled — no device needed
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer())
    stt = DeepgramSTTService(api_key=Config.ASR_API_KEY or "dummy", model=Config.ASR_MODEL)
    llm = OpenAILLMService(api_key=Config.OPENROUTER_API_KEY or "dummy",
                           base_url=Config.OPENROUTER_BASE_URL,
                           settings=OpenAILLMService.Settings(model=Config.FAST_MODEL))
    tts = ElevenLabsTTSService(api_key=Config.TTS_API_KEY or "dummy",
                               settings=ElevenLabsTTSService.Settings(
                                   voice=Config.TTS_VOICE_ID, model=Config.TTS_MODEL))

    async def submit_task(params: FunctionCallParams, description: str) -> str:
        """Delegate a task to the background coding agent.

        Args:
            description: What the background agent should build or do.
        """
        return "task-123"

    context = LLMContext([{"role": "system", "content": "hi"}], tools=[submit_task])
    agg = LLMContextAggregatorPair(context)
    pipeline = Pipeline([transport.input(), stt, vad, agg.user(), llm, tts, agg.assistant(), transport.output()])
    worker = PipelineWorker(pipeline)
    schemas = [w.to_function_schema() for w in context._tools.direct_functions]
    for s in schemas:
        print("tool:", s.name, "props:", list(s.properties), "required:", s.required)
    print("PIPELINE OK (worker + tools registered)")


if __name__ == "__main__":
    asyncio.run(main())