import logging

from anthropic import AsyncAnthropic

import config

logger = logging.getLogger(__name__)

client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)


class LLMManager:
    """Per-call conversation state plus streaming response generation.

    History writes are owned by the caller (main.py), which knows whether a
    stream finished completely or was interrupted mid-way.
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

    async def generate_response(self, call_id: str):
        """Async generator of response tokens. Does NOT mutate history."""
        session = self.ensure_session(call_id)
        try:
            async with client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=256,
                system=config.SYSTEM_PROMPT,
                messages=session,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception:
            logger.exception("LLM error for call %s", call_id)
            yield "I'm sorry, I'm having trouble thinking right now."


llm_manager = LLMManager()
