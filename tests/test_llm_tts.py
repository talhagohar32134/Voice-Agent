import asyncio

import llm
import tts


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------
# LLM session management
# ------------------------------------------------------------------

def test_outbound_session_seeded_with_greeting():
    mgr = llm.LLMManager()
    session = mgr.ensure_session("call-1", is_outbound=True)
    assert len(session) == 1
    assert session[0]["role"] == "assistant"
    assert "greeting" in session[0]["content"].lower() or session[0]["content"]


def test_inbound_session_starts_empty():
    mgr = llm.LLMManager()
    session = mgr.ensure_session("call-2", is_outbound=False)
    assert session == []


def test_user_and_assistant_messages_append():
    mgr = llm.LLMManager()
    mgr.add_user_message("call-3", "hello")
    mgr.add_assistant_message("call-3", "hi there")
    mgr.add_assistant_message("call-3", "   ")  # blank must be ignored
    session = mgr.get_or_create_session("call-3")
    roles = [m["role"] for m in session]
    assert roles == ["user", "assistant"]


def test_end_session_frees_memory():
    mgr = llm.LLMManager()
    mgr.add_user_message("call-4", "hi")
    mgr.end_session("call-4")
    assert "call-4" not in mgr.sessions
    # Safe to call again / re-created empty
    assert mgr.get_or_create_session("call-4") == []


def test_generate_response_is_generator_without_history_mutation(monkeypatch):
    """Streaming must yield tokens but NOT touch history (main.py owns that)."""

    class FakeMessageStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @property
        def text_stream(self):
            async def _gen():
                yield "one "
                yield "two"

            return _gen()

    class FakeMessages:
        def stream(self, **kwargs):
            return FakeMessageStream()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr(llm, "client", FakeClient())

    mgr = llm.LLMManager()
    mgr.add_user_message("call-5", "say something")

    async def collect():
        return [tok async for tok in mgr.generate_response("call-5")]

    tokens = _run(collect())
    assert "".join(tokens) == "one two"
    # History unchanged - only the user message we added ourselves
    assert [m["role"] for m in mgr.get_or_create_session("call-5")] == ["user"]


# ------------------------------------------------------------------
# TTS sentence chunking
# ------------------------------------------------------------------

class FakeConvert:
    def __init__(self):
        self.spoken_texts = []

    async def __call__(self, *, voice_id, text, **kwargs):
        self.spoken_texts.append(text)

        async def _chunks():
            for b in (b"\x01", b"\x02"):
                yield b

        return _chunks()


class FakeTextToSpeech:
    def __init__(self):
        self.convert = FakeConvert()


class FakeClient:
    def __init__(self):
        self.text_to_speech = FakeTextToSpeech()


async def _tokens(*words):
    for w in words:
        yield w


def test_sentences_are_spoken_as_completed():
    fake_client = FakeClient()
    tts.client = fake_client

    async def consume():
        out = [
            chunk
            async for chunk in tts.generate_audio_stream(
                _tokens("Hello there. ", "How can I help", " you today?")
            )
        ]
        return out

    chunks = _run(consume())
    spoken = fake_client.text_to_speech.convert.spoken_texts
    assert spoken == ["Hello there.", "How can I help you today?"]
    assert all(isinstance(c, bytes) for c in chunks)
    assert len(chunks) == 4  # 2 audio bytes x 2 sentences


def test_final_partial_sentence_is_flushed():
    fake_client = FakeClient()
    tts.client = fake_client

    async def consume():
        async for _ in tts.generate_audio_stream(_tokens("Just a trailing fragment")):
            pass

    _run(consume())
    spoken = fake_client.text_to_speech.convert.spoken_texts
    assert spoken == ["Just a trailing fragment"]
