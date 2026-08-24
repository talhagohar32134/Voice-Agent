import asyncio
import json
import logging
from datetime import datetime

from anthropic import AsyncAnthropic

import config
import scheduler

logger = logging.getLogger(__name__)


def _system_prompt() -> str:
    """System prompt plus live clock so relative dates ('tomorrow') resolve."""
    now = datetime.now().strftime("%A %d %B %Y, %I:%M %p")
    return (
        f"{config.SYSTEM_PROMPT}\n"
        f"Current date and time: {now}. Resolve words like 'today' or 'tomorrow' "
        "against this before calling tools.\n"
        "LANGUAGE RULE (critical): Mirror the caller's language in EVERY reply. "
        "If the caller's last message contains Devanagari script, your ENTIRE reply "
        "MUST be in Hindi (Devanagari script). If it contains Arabic-script words "
        "(Urdu), your ENTIRE reply MUST be in Urdu (Arabic script). If they used "
        "Roman Urdu/Hinglish, reply in Roman Urdu. Use plain English only when their "
        "message was fully English. Switch automatically whenever they switch - like "
        "a human agent would. Never comment on or translate the language."
    )

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

MAX_TOOL_ROUNDS = 3  # guard against runaway tool loops


class LLMManager:
    """Per-call conversation state plus streaming response generation.

    History writes are owned by the caller (main.py), which knows whether a
    stream finished completely or was interrupted mid-way.

    Providers:
      - anthropic (Claude) via native SDK - plain chat, no tools yet
      - groq (free tier) via OpenAI-compatible API - WITH function calling
    """

    def __init__(self):
        self.sessions = {}

    def get_or_create_session(self, call_id: str):
        return self.sessions.setdefault(call_id, [])

    def ensure_session(self, call_id: str, is_outbound: bool = False):
        if call_id and call_id not in self.sessions:
            self.sessions[call_id] = []
            if is_outbound:
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
        self.sessions.pop(call_id, None)

    # ------------------------------------------------------------------
    # Anthropic (plain chat)
    # ------------------------------------------------------------------

    async def _stream_anthropic(self, session, extra_system: str = ""):
        system = _system_prompt() + (f"\n{extra_system}" if extra_system else "")
        async with _anthropic_client.messages.stream(
            model=config.ANTHROPIC_MODEL,
            max_tokens=256,
            system=system,
            messages=session,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # Groq with function calling
    # ------------------------------------------------------------------

    async def _groq_request_stream(self, messages):
        return await _groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            max_tokens=300,
            stream=True,
            tools=scheduler.TOOLS_SCHEMA,
        )

    async def _stream_groq_with_tools(self, call_id: str, session, extra_system: str = ""):
        """Stream a reply; transparently execute tool calls and continue.

        The caller sees only natural-language tokens. Tool traffic is
        appended to history so the model keeps context across rounds.
        """
        system = _system_prompt() + (f"\n{extra_system}" if extra_system else "")
        messages = [{"role": "system", "content": system}] + session
        context = {"phone_number": "", "call_id": call_id}

        for _round in range(MAX_TOOL_ROUNDS):
            content_parts = []
            tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}

            response = await self._groq_request_stream(messages)
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
                    yield delta.content  # speak through immediately
                for tc in delta.tool_calls or []:
                    slot = tool_calls.setdefault(tc.index, {"id": tc.id, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments

            if not tool_calls:
                break  # pure text answer - done

            # Persist assistant turn that requested the tools, then run them.
            assistant_msg = {
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for _idx, tc in sorted(tool_calls.items())
                ],
            }
            messages.append(assistant_msg)
            session.append(assistant_msg)

            for _, tc in sorted(tool_calls.items()):
                logger.info("Tool call %s(%s) for call %s", tc["name"], tc["arguments"], call_id)
                result = await asyncio.to_thread(scheduler.execute_tool, tc["name"], tc["arguments"], context)
                logger.info("Tool result: %s", result[:200])
                tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result}
                messages.append(tool_msg)
                session.append(tool_msg)
            # loop continues -> model now answers with real tool data
        else:
            logger.warning("Tool round limit hit for call %s", call_id)

    async def generate_response(self, call_id: str, extra_system: str = ""):
        """Async generator of response tokens. Does NOT mutate history
        EXCEPT tool-call/tool-result bookkeeping required by the API."""
        session = self.ensure_session(call_id)
        try:
            if config.LLM_PROVIDER == "groq":
                stream = self._stream_groq_with_tools(call_id, session, extra_system)
            else:
                stream = self._stream_anthropic(session, extra_system)
            async for text in stream:
                yield text
        except Exception:
            logger.exception("LLM error (%s) for call %s", config.LLM_PROVIDER, call_id)
            yield "I'm sorry, I'm having trouble thinking right now."


_MOOD_WORDS = {"positive", "neutral", "worried", "frustrated", "angry"}

# Fast deterministic pass - catches obvious emotional language instantly
_KEYWORD_MOODS = {
    "angry": ["angry", "furious", "ridiculous", "unacceptable", "outrageous",
              "worst", "useless", "pathetic", "gussa", "naraz"],
    "frustrated": ["frustrated", "frustrating", "nobody called", "waiting for weeks",
                   "weeks now", "again and again", "sick of", "tired of", "complaint",
                   "no response", "not helpful", "pareshan"],
    "worried": ["worried", "scared", "anxious", "nervous", "afraid", "concerned",
                "serious problem", "is everything ok", "ghabrahat"],
    "positive": ["thank you", "thanks", "great", "perfect", "wonderful", "happy",
                 "appreciate", "awesome", "shukriya", "bahut acha"],
}
_KEYWORD_PRIORITY = ["angry", "frustrated", "worried", "positive"]

_EMPATHY_HINTS = {
    "worried": "The caller sounds WORRIED. Reassure them calmly and clearly before anything else.",
    "frustrated": "The caller is FRUSTRATED. Acknowledge their annoyance sincerely, apologise briefly, and be extra helpful.",
    "angry": "The caller is ANGRY. Stay very calm and warm, apologise once, do not argue, focus on fixing their problem fast.",
}


def _keyword_mood(text: str) -> str | None:
    low = f" {text.lower()} "
    best = None
    for mood in _KEYWORD_PRIORITY:
        if any(kw in low for kw in _KEYWORD_MOODS[mood]):
            return mood  # priority order wins
    return best


async def detect_mood(text: str) -> str:
    """Classify caller mood. Keyword fast-path first, then a small LLM call."""
    kw = _keyword_mood(text)
    if kw:
        return kw
    if config.LLM_PROVIDER != "groq" or _groq_client is None:
        return "neutral"
    try:
        resp = await _groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the caller's emotional state from this phone "
                        "transcript. Answer with EXACTLY one word: positive, "
                        "neutral, worried, frustrated, or angry."
                    ),
                },
                {"role": "user", "content": text[:500]},
            ],
            max_tokens=40,
            temperature=0,
        )
        word = (resp.choices[0].message.content or "").strip().lower().split()
        mood = word[-1] if word else "neutral"
        return mood.strip(".,!") if mood.strip(".,!") in _MOOD_WORDS else "neutral"
    except Exception:
        logger.exception("Mood detection failed")
        return "neutral"


llm_manager = LLMManager()
