import pytest

from agent.interruption import InterruptionTracker


def test_interruption_tracker_records_latency():
    t = InterruptionTracker()
    t.on_bot_started_speaking()
    t.on_user_started_speaking()          # barge-in while bot speaks
    t.on_user_stopped_speaking()
    assert len(t.events) == 1
    ev = t.events[0]
    assert ev.latency >= 0
    # a second bot speech start without interruption must not add events
    t.on_bot_started_speaking()
    t.on_user_started_speaking()
    t.on_user_stopped_speaking()
    assert len(t.events) == 2


def test_tracker_no_event_when_user_speaks_first():
    t = InterruptionTracker()
    t.on_user_started_speaking()
    t.on_user_stopped_speaking()
    assert t.events == []
    assert t.last_speech_onset > 0
