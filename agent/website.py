"""Website build workflow (Phase 3): generate files, self-check, and report a preview path.

The build itself is delegated to the background coding agent as a normal task, so it never
blocks the conversational agent. Publishing is a separate, approval-gated operation
(services.github_client.publish_website).
"""
from agent.config import Config


def build_website_task(request: str) -> str:
    """Return the prompt sent to the background coding agent."""
    ws = Config.WORKSPACE_DIR
    return (
        "Create a small static website in the workspace subdirectory 'site':\n"
        f"{request}\n"
        "Requirements:\n"
        "- index.html plus styles.css (and script.js if needed)\n"
        "- clean, modern, responsive; semantic HTML\n"
        "- after writing the files, verify index.html exists and contains the requested content\n"
        "- reply with a one-paragraph summary of what you built"
    )


def preview_url() -> str:
    """Local preview instructions (static site — just open the file, or serve the dir)."""
    return f"file://{Config.WORKSPACE_DIR / 'site' / 'index.html'}"
