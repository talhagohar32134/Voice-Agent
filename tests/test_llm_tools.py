"""Tool-calling loop tests with a fake Groq client."""

import asyncio

import llm


def _run(coro):
    return asyncio.run(coro)


class _Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _text_chunk(text):
    from types import SimpleNamespace

    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))])


def _tool_call_chunk(name, arguments, index=0, call_id="call_1"):
    from types import SimpleNamespace

    fn = SimpleNamespace(name=name, arguments=arguments)
    tc = SimpleNamespace(id=call_id, index=index, function=fn)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))]
    )


class FakeCompletions:
    def __init__(self, scripted):
        # list of (stream_chunks) returned per create() call
        self.scripted = list(scripted)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Stream(self.scripted.pop(0))


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeGroq:
    def __init__(self, completions):
        self.chat = FakeChat(completions)


def test_tool_call_roundtrip(monkeypatch):
    """LLM asks for a tool -> tool runs -> model answers using the result."""
    fake = FakeCompletions(
        [
            [_tool_call_chunk("check_availability", '{"date": "2026-09-01"}')],
            [_text_chunk("We have "), _text_chunk("9 AM open.")],
        ]
    )
    monkeypatch.setattr(llm, "_groq_client", FakeGroq(fake))
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "groq")

    executed = {}
    monkeypatch.setattr(
        llm.scheduler,
        "execute_tool",
        lambda name, args, ctx: executed.update({"name": name, "args": args}) or "Open slots: 09:00",
    )

    mgr = llm.LLMManager()
    mgr.add_user_message("call-t1", "book me something")

    async def collect():
        return [tok async for tok in mgr.generate_response("call-t1")]

    tokens = _run(collect())
    assert "".join(tokens) == "We have 9 AM open."
    assert executed == {"name": "check_availability", "args": '{"date": "2026-09-01"}'}

    session = mgr.get_or_create_session("call-t1")
    roles = [m["role"] for m in session]
    # user -> assistant(tool_calls) -> tool(result); final text is spoken but NOT
    # appended here (main.py owns that write)
    assert roles == ["user", "assistant", "tool"]
    assert session[-1]["content"] == "Open slots: 09:00"

    # Second request must include tools + full history
    second_call = fake.calls[1]
    assert second_call["tools"]  # schema present
    assert any(m.get("role") == "tool" for m in second_call["messages"])


def test_pure_text_answer_skips_tools(monkeypatch):
    fake = FakeCompletions([[_text_chunk("Hello! "), _text_chunk("How can I help?")]])
    monkeypatch.setattr(llm, "_groq_client", FakeGroq(fake))
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "groq")

    mgr = llm.LLMManager()
    mgr.add_user_message("call-t2", "hi")

    async def collect():
        return [tok async for tok in mgr.generate_response("call-t2")]

    tokens = _run(collect())
    assert "".join(tokens) == "Hello! How can I help?"
    # No tool bookkeeping added
    assert [m["role"] for m in mgr.get_or_create_session("call-t2")] == ["user"]


def test_tool_round_limit_guard(monkeypatch):
    """Model keeps calling tools forever -> we cut off after MAX_TOOL_ROUNDS."""
    endless = [[_tool_call_chunk("list_appointments", "{}", call_id=f"c{i}")] for i in range(10)]
    fake = FakeCompletions(endless)
    monkeypatch.setattr(llm, "_groq_client", FakeGroq(fake))
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(llm.scheduler, "execute_tool", lambda *a, **k: "none found")

    mgr = llm.LLMManager()
    mgr.add_user_message("call-t3", "loop please")

    async def collect():
        return [tok async for tok in mgr.generate_response("call-t3")]

    tokens = _run(collect())
    assert "".join(tokens) == ""  # no text ever came - guard stopped the loop
    assert len(fake.calls) <= llm.MAX_TOOL_ROUNDS
