"""Minimal test: does a worker-registered observer actually receive pushed frames?"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.observability import DashboardObserver, EventBus


class Source:
    """Pushes a few frames then stops."""
    pass


async def main() -> None:
    from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame, LLMTextFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.workers.runner import WorkerRunner

    bus = EventBus()
    obs = DashboardObserver(bus)

    class Pusher(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await self.push_frame(frame, direction)

    class Generator(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            print("Generator saw:", type(frame).__name__, flush=True)
            if isinstance(frame, TextFrame):
                await self.push_frame(TranscriptionFrame("hello there", "", 16000))
                await self.push_frame(LLMTextFrame("hi!"))
                await self.push_frame(TextFrame("__done__"))

    pipeline = Pipeline([Generator(), Pusher()])
    worker = PipelineWorker(pipeline, observers=[obs.pipecat_observer()])

    async def drive():
        await asyncio.sleep(2.5)
        await worker.queue_frame(TextFrame("hello there"))
        await asyncio.sleep(2.5)
        await runner.cancel()

    runner = WorkerRunner()
    await runner.add_workers(worker)
    t = asyncio.create_task(drive())
    await runner.run()
    t.cancel()

    kinds = [e["type"] for e in bus._history]
    print("constructor-observer events:", kinds)

    # try post-construction registration instead
    bus2 = EventBus()
    obs2 = DashboardObserver(bus2)
    worker2 = PipelineWorker(Pipeline([Generator(), Pusher()]))
    worker2.add_observer(obs2.pipecat_observer())
    worker2_params = None

    async def drive2():
        await asyncio.sleep(2.5)
        await worker2.queue_frame(TextFrame("hello there"))
        await asyncio.sleep(2.5)
        await runner2.cancel()

    runner2 = WorkerRunner()
    await runner2.add_workers(worker2)
    t2 = asyncio.create_task(drive2())
    await runner2.run()
    t2.cancel()
    kinds2 = [e["type"] for e in bus2._history]
    print("post-construction events:", kinds2)
    assert "transcript" in kinds2 or "transcript" in kinds, "OBSERVER NOT RECEIVING FRAMES"
    print("OBSERVER OK")


if __name__ == "__main__":
    asyncio.run(main())