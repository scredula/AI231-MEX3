"""TTSTextNormalizer: markdown must never reach the TTS engine.

The TTS reads symbols literally ("**" -> "star star"). The normalizer sits between
LLM and TTS and strips markdown from both TextFrame and TTSSpeakFrame content.
"""

import asyncio

from agent.pipeline import TTSTextNormalizer
from pipecat.frames.frames import TextFrame, TTSSpeakFrame


def _clean(text):
    return TTSTextNormalizer().clean(text)


def _run_through(frame):
    norm = TTSTextNormalizer()
    seen = []
    norm.push_frame = lambda f, d: seen.append(f)  # capture output
    asyncio.get_event_loop().run_until_complete(norm.process_frame(frame, None))
    return seen[0]


def test_bold_is_stripped():
    assert _clean("**Hello** world") == "Hello world"
    assert _clean("a **bold** and **more**") == "a bold and more"


def test_italic_and_underline():
    assert _clean("*hello* world") == "hello world"
    assert _clean("_hello_ world") == "hello world"


def test_code_and_strikethrough():
    assert _clean("run `ls -la` now") == "run ls -la now"
    assert _clean("~~old~~ new") == "old new"


def test_headings_and_bullets_removed():
    assert _clean("## Results\n- item one\n- item two") == "Results\nitem one\nitem two"
    assert _clean("* bullet") == "bullet"


def test_markdown_link_becomes_text():
    assert _clean("see [the docs](https://example.com/x) for more") == "see the docs for more"


def test_bare_url_becomes_domain():
    assert _clean("open https://www.example.com/page") == "open example.com/page"


def test_text_frame_normalized_in_place():
    frame = _run_through(TextFrame("Hello **bold**"))
    assert frame.text == "Hello bold"


def test_tts_speak_frame_normalized_in_place():
    frame = _run_through(TTSSpeakFrame("Task **done**", append_to_context=False))
    assert frame.text == "Task done"


def test_plain_prose_untouched():
    text = "It costs 3 dollars and 4 cents, or about 50%."
    assert _clean(text) == text


def test_math_symbols_not_eaten():
    # asterisks mid-sentence (e.g. multiplication) must not be touched
    assert _clean("2*3 is 6") == "2*3 is 6"
