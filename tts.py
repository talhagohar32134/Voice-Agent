import logging
import re

import httpx

import config

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


# ------------------------------------------------------------------
# Language handling - human-style code switching
# ------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Detect script-based language: Urdu (Arabic script), Hindi (Devanagari), else English/Roman."""
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F")
    devanagari = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")

    if arabic >= 2 and arabic >= devanagari:
        return "ur"
    if devanagari >= 2:
        return "hi"
    return "en"


def _voice_for(lang: str) -> str:
    return {
        "ur": config.EDGE_VOICE_UR,
        "hi": config.EDGE_VOICE_HI,
    }.get(lang, config.EDGE_VOICE_EN)


def _pcm_to_ulaw_chunks(samples, chunk_samples=800):
    """16-bit PCM ints -> ulaw bytes, yielded in ~100ms blocks at 8kHz."""
    BIAS, CLIP = 0x84, 32635

    def linear_to_ulaw(v: int) -> int:
        x = max(-CLIP, min(CLIP, v))
        sign = (x >> 8) & 0x80
        if sign:
            x = -x
        x += BIAS
        exp, mask = 7, 0x4000
        while exp > 0 and not (x & mask):
            exp -= 1
            mask >>= 1
        mant = (x >> (exp + 3)) & 0x0F
        return ~(sign | (exp << 4) | mant) & 0xFF

    out = bytearray()
    for s in samples:
        out.append(linear_to_ulaw(s))
        if len(out) >= chunk_samples:
            yield bytes(out)
            out.clear()
    if out:
        yield bytes(out)


async def _mp3_to_ulaw(mp3_bytes: bytes):
    """Decode mp3 (any rate/channels) -> mono 8kHz ulaw chunks."""
    import miniaudio

    decoded = miniaudio.decode(mp3_bytes)
    samples = list(decoded.samples)
    channels = max(1, decoded.nchannels)

    # Downmix to mono
    if channels > 1:
        frames = len(samples) // channels
        samples = [sum(samples[i * channels:(i + 1) * channels]) // channels for i in range(frames)]

    # Resample to 8kHz by window averaging
    ratio = max(1, round(decoded.sample_rate / 8000))
    if ratio > 1:
        resampled = []
        for i in range(0, len(samples) - ratio + 1, ratio):
            window = samples[i:i + ratio]
            resampled.append(sum(window) // len(window))
        samples = resampled

    for chunk in _pcm_to_ulaw_chunks(samples):
        yield chunk


# ------------------------------------------------------------------
# Provider: ElevenLabs
# ------------------------------------------------------------------

_elevenlabs_client = None
if config.TTS_PROVIDER == "elevenlabs":
    from elevenlabs.client import AsyncElevenLabs

    _elevenlabs_client = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY or "missing")


async def _speak_sentence_elevenlabs(sentence: str):
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


# ------------------------------------------------------------------
# Provider: Microsoft Edge Neural TTS (free, no key, multilingual)
# ------------------------------------------------------------------

async def _speak_sentence_edge(sentence: str, lang: str = "en"):
    import edge_tts

    voice = _voice_for(lang)
    communicate = edge_tts.Communicate(sentence, voice)
    mp3 = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    logger.debug("edge-tts %s: %d chars -> %d bytes mp3", voice, len(sentence), len(mp3))
    async for ulaw_chunk in _mp3_to_ulaw(bytes(mp3)):
        yield ulaw_chunk


# ------------------------------------------------------------------
# Router: pick the best provider per sentence
# ------------------------------------------------------------------

async def _speak_sentence(sentence: str):
    lang = detect_language(sentence)

    # Non-English always goes to Edge (Aura/ElevenLabs configs here are English-only).
    if lang != "en" or config.TTS_PROVIDER == "edge":
        gen = _speak_sentence_edge(sentence, lang)
    elif config.TTS_PROVIDER == "elevenlabs":
        gen = _speak_sentence_elevenlabs(sentence)
    else:
        gen = _speak_sentence_deepgram(sentence)

    async for chunk in gen:
        yield chunk


async def generate_audio_stream(text_stream):
    """
    Takes an async generator yielding text tokens and yields ulaw_8000 audio
    chunks for Twilio media streams.

    Sentences are synthesized as they complete; the language of each sentence
    picks the voice automatically, so mid-call English<->Urdu/Hindi switches
    just work.
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
