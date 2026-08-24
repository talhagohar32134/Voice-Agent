"""Mood detection + dashboard tests."""

import asyncio
from types import SimpleNamespace

import llm
from database import Transcript


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------
# Mood classification (fake Groq)
# ------------------------------------------------------------------

class _Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def test_detect_mood_returns_valid_word(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            assert kwargs["max_tokens"] >= 30  # room for reasoning models
            msg = SimpleNamespace(content="frustrated")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class FakeChat:
        completions = FakeCompletions()

    monkeypatch.setattr(llm, "_groq_client", SimpleNamespace(chat=FakeChat()))
    # No keyword hits -> LLM fallback decides
    mood = _run(llm.detect_mood("I really need help sorting my billing"))
    assert mood == "frustrated"


def test_detect_mood_garbage_reply_falls_back(monkeypatch):
    from types import SimpleNamespace

    class FakeCompletions:
        async def create(self, **kwargs):
            msg = SimpleNamespace(content="I cannot classify that")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class FakeChat:
        completions = FakeCompletions()

    monkeypatch.setattr(llm, "_groq_client", SimpleNamespace(chat=FakeChat()))
    assert _run(llm.detect_mood("hello")) == "neutral"


def test_keyword_fast_path_beats_llm(monkeypatch):
    """Obvious frustration must be classified instantly, no LLM call needed."""
    calls = {"n": 0}

    class FakeCompletions:
        async def create(self, **kwargs):
            calls["n"] += 1
            raise AssertionError("LLM should not be called for keyword matches")

    class FakeChat:
        completions = FakeCompletions()

    monkeypatch.setattr(llm, "_groq_client", SimpleNamespace(chat=FakeChat()))
    assert _run(llm.detect_mood("this is absolutely ridiculous")) == "angry"
    assert _run(llm.detect_mood("I am so frustrated with this service")) == "frustrated"
    assert _run(llm.detect_mood("I am worried about my results")) == "worried"
    assert _run(llm.detect_mood("perfect thank you so much")) == "positive"
    assert calls["n"] == 0


def test_priority_anger_over_frustration():
    assert llm._keyword_mood("ridiculous and I am tired of this") == "angry"


# ------------------------------------------------------------------
# Dashboard endpoints
# ------------------------------------------------------------------

def test_calls_list_and_transcript(api_client, db):
    db.add(Transcript(call_id="CALL-X", role="user", text="mujhe appointment chahiye", mood="neutral"))
    db.add(Transcript(call_id="CALL-X", role="agent", text="Ji bilkul, kab chahiye?"))
    db.add(Transcript(call_id="CALL-Y", role="user", text="you people never help", mood="angry"))
    db.commit()

    res = api_client.get("/calls")
    assert res.status_code == 200
    calls = {c["call_id"]: c for c in res.json()}
    assert calls["CALL-X"]["messages"] == 2
    assert calls["CALL-X"]["dominant_mood"] == "neutral"
    assert calls["CALL-Y"]["dominant_mood"] == "angry"

    res2 = api_client.get("/calls/CALL-X/transcript")
    msgs = res2.json()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["mood"] == "neutral"
    assert msgs[1]["role"] == "agent"


def test_empty_dashboard_is_fine(api_client):
    res = api_client.get("/calls")
    assert res.status_code == 200
    assert res.json() == []


def test_dashboard_page_served(api_client):
    res = api_client.get("/dashboard")
    assert res.status_code == 200
    assert "Conversation Dashboard" in res.text
