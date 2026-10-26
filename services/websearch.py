"""Web search via DuckDuckGo (ddgs package) — free, no API key, pure Python (ARM64-ok)."""
import logging

log = logging.getLogger("websearch")


def search_web(query: str, max_results: int = 4) -> str:
    """Run a web search and return compact text results (title, snippet, URL)."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "error: search backend not installed (pip install ddgs)"
    if not query or len(query) > 400:
        return "error: query must be 1..400 characters"
    try:
        rows = DDGS().text(query, max_results=max_results)
    except Exception as e:
        log.warning("web search failed: %s", e)
        return f"error: search failed ({type(e).__name__})"
    if not rows:
        return "no results"
    lines = []
    for r in rows:
        title = (r.get("title") or "")[:120]
        body = (r.get("body") or "")[:220]
        href = r.get("href") or r.get("url") or ""
        lines.append(f"- {title} | {body} | {href}")
    return "\n".join(lines)