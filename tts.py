import logging
import re

import httpx

import config

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# ------------------------------------------------------------------
# Provider: ElevenLabs
# ------------------------------------------------------------------

_elevenlabs_client = None
if config.TTS_PROVIDER == "elevenlabs":
    from elevenlabs.client import AsyncElevenLabs

    _elevenlabs_client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY or "missing")


async def _speak_sentence_elevenlabs(sentence: str):
    """Yield ulaw_8000 audio chunks for one sentence via ElevenLabs."""
    audio = await _elevenlabs_client.text_to_speech.convert(
        voice_id=config.ELEVENLABS_VOICE_ID,
        text=sentence,
        model_id="eleven_turbo_v2_5",
        output_format="ulaw_8000",
        optimize_streaming_latency=3,
    )
    async for chunk in audio:
        yield chunk


# ------------------------------------------------------------------
# Provider: Deepgram Aura (shares the $200 free STT credit)
# ------------------------------------------------------------------

_deepgram_http: httpx.AsyncClient | None = None


def _get_deepgram_http() -> httpx.AsyncClient:
    global _deepgram_http
    if _deepgram_http is None or _deepgram_http.is_closed:
        _deepgram_http = httpx.AsyncClient(
            headers={
                "Authorization": f"Token {config.DEEPGRAM_API_KEY or 'missing'}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
    return _deepgram_http


async def _speak_sentence_deepgram(sentence: str):
    """Yield ulaw_8000 audio chunks for one sentence via Deepgram Aura.

    container=none returns raw mulaw bytes - exactly what Twilio media
    streams expect.
    """
    url = (
        "https://api.deepgram.com/v1/speak"
        f"?model={config.DEEPGRAM_TTS_MODEL}&encoding=mulaw&sample_rate=8000&container=none"
    )
    client = _get_deepgram_http()
    async with client.stream("POST", url, json={"text": sentence}) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk


async def _speak_sentence(sentence: str):
    if config.TTS_PROVIDER == "deepgram":
        gen = _speak_sentence_deepgram(sentence)
    else:
        gen = _speak_sentence_elevenlabs(sentence)
    async for chunk in gen:
        yield chunk


async def generate_audio_stream(text_stream):
    """
    Takes an async generator yielding text chunks (from Claude/Groq) and returns
    an async generator yielding audio chunks in ulaw_8000 format for Twilio.

    Neither provider accepts a live text stream on this endpoint set, so we
    buffer into sentences and synthesize each sentence as it completes. First
    audio goes out after the first full sentence - natural prosody, low latency.
    """
    buffer = ""

    async for token in text_stream:
        buffer += token
        while True:
            match = _SENTENCE_BOUNDARY.search(buffer)
            if not match:
                break
            sentence = buffer[: match.end()].strip()
            buffer = buffer[match.end():]
            if sentence:
                try:
                    async for chunk in _speak_sentence(sentence):
                        yield chunk
                except Exception:
                    logger.exception("TTS error for sentence: %r", sentence[:80])

    if buffer.strip():
        try:
            async for chunk in _speak_sentence(buffer.strip()):
                yield chunk
        except Exception:
            logger.exception("TTS error for final chunk: %r", buffer[:80])
