import logging

from anthropic import AsyncAnthropic

import config

logger = logging.getLogger(__name__)

_anthropic_client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY or "missing")

_groq_client = None
if config.LLM_PROVIDER == "groq":
    try:
        from openai import AsyncOpenAI

        _groq_client = AsyncOpenAI(
            api_key=config.GROQ_API_KEY or "missing",
            base_url="https://api.groq.com/openai/v1",
        )
    except ImportError:
        logger.warning("openai package not installed - Groq provider unavailable")


class LLMManager:
    """Per-call conversation state plus streaming response generation.

    History writes are owned by the caller (main.py), which knows whether a
    stream finished completely or was interrupted mid-way.

    Providers:
      - anthropic (Claude) via native SDK
      - groq (free tier) via OpenAI-compatible API
    """

    def __init__(self):
        self.sessions = {}

    def get_or_create_session(self, call_id: str):
        return self.sessions.setdefault(call_id, [])

    def ensure_session(self, call_id: str, is_outbound: bool = False):
        """Create session on call start so outbound calls get their greeting."""
        if call_id and call_id not in self.sessions:
            self.sessions[call_id] = []
            if is_outbound:
                # Agent "speaks" first on outbound - seed history with greeting.
                self.sessions[call_id].append(
                    {"role": "assistant", "content": config.OUTBOUND_GREETING}
                )
        return self.sessions[call_id]

    def add_user_message(self, call_id: str, text: str):
        self.ensure_session(call_id).append({"role": "user", "content": text})

    def add_assistant_message(self, call_id: str, text: str):
        if text.strip():
            self.ensure_session(call_id).append({"role": "assistant", "content": text.strip()})

    def end_session(self, call_id: str):
        """Free memory once a call is over."""
        self.sessions.pop(call_id, None)

    async def _stream_anthropic(self, session):
        async with _anthropic_client.messages.stream(
            model=config.ANTHROPIC_MODEL,
            max_tokens=256,
            system=config.SYSTEM_PROMPT,
            messages=session,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_groq(self, session):
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + session
        response = await _groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            max_tokens=256,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def generate_response(self, call_id: str):
        """Async generator of response tokens. Does NOT mutate history."""
        session = self.ensure_session(call_id)
        try:
            if config.LLM_PROVIDER == "groq":
                stream = self._stream_groq(session)
            else:
                stream = self._stream_anthropic(session)
            async for text in stream:
                yield text
        except Exception:
            logger.exception("LLM error (%s) for call %s", config.LLM_PROVIDER, call_id)
            yield "I'm sorry, I'm having trouble thinking right now."


llm_manager = LLMManager()
