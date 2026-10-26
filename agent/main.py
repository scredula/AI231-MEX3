"""Entry point.

  python -m agent.main            # text mode (no audio needed — dev/test/demo)
  python -m agent.main --voice    # Pipecat voice pipeline (mic + speaker)

Voice mode imports the audio stack lazily so text mode runs on any machine.
"""
import argparse
import asyncio
import logging
import sys

from agent.config import Config


def _setup_logging():
    level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(level=level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")


def _make_coordinator():
    from agent.coordinator import Coordinator, TaskStore
    from coding_agent.runner import make_runner_factory
    Config.ensure_dirs()
    store = TaskStore(Config.TASKS_DB_PATH)
    return Coordinator(store, make_runner_factory(Config.WORKSPACE_DIR))


async def run_text_mode() -> None:
    from agent.conversation import ConversationAgent
    coordinator = _make_coordinator()
    coordinator.on_announce(lambda t: print(f"[announce] {t}"))
    agent = ConversationAgent(coordinator)
    print("Text-mode agent ready. Try: 'build me a portfolio website' / 'status <id>' / 'quit'.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        print("agent>", await agent.reply(text))


async def run_voice_mode() -> None:
    from agent.pipeline import VoicePipelineAgent
    coordinator = _make_coordinator()
    await VoicePipelineAgent(coordinator).run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-duplex voice AI agent (AI231-MEX3)")
    parser.add_argument("--voice", action="store_true", help="run the Pipecat voice pipeline")
    args = parser.parse_args()
    _setup_logging()
    asyncio.run(run_voice_mode() if args.voice else run_text_mode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
