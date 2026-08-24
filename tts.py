import logging
import re

from elevenlabs.client import AsyncElevenLabs

import config

logger = logging.getLogger(__name__)

client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


async def _speak_sentence(sentence: str):
    """Yield ulaw_8000 audio chunks for one sentence."""
    audio = await client.text_to_speech.convert(
        voice_id=config.ELEVENLABS_VOICE_ID,
        text=sentence,
        model_id="eleven_turbo_v2_5",
        output_format="ulaw_8000",
        optimize_streaming_latency=3,
    )
    async for chunk in audio:
        yield chunk


async def generate_audio_stream(text_stream):
    """
    Takes an async generator yielding text chunks (from Claude) and returns an
    async generator yielding audio chunks in ulaw_8000 format for Twilio.

    The ElevenLabs v2 SDK has no streaming-text-input endpoint, so we buffer
    into sentences and synthesize each sentence as it completes. First audio
    goes out after the first full sentence - natural prosody, low latency.
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
