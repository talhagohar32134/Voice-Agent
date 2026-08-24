import asyncio
import sys
from types import SimpleNamespace

import tts


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------
# Language detection (script based)
# ------------------------------------------------------------------

def test_english_latin_detected():
    assert tts.detect_language("I would like to book an appointment") == "en"


def test_urdu_arabic_script_detected():
    assert tts.detect_language("مجھے کل کا اپائنٹمنٹ چاہیے") == "ur"


def test_hindi_devanagari_detected():
    assert tts.detect_language("मुझे कल अपॉइंटमेंट चाहिए") == "hi"


def test_single_stray_char_still_english():
    # One stray glyph should not flip the whole sentence
    assert tts.detect_language("ok a") == "en"


def test_voice_selection_per_language():
    assert tts._voice_for("en") == "en-US-AvaNeural"
    assert tts._voice_for("ur") == "ur-PK-AsadNeural"
    assert tts._voice_for("hi") == "hi-IN-SwaraNeural"


# ------------------------------------------------------------------
# Edge TTS pipeline (mocked network + decoder)
# ------------------------------------------------------------------

def test_edge_pipeline_converts_mp3_to_ulaw(monkeypatch):
    captured_voices = []

    class FakeCommunicate:
        def __init__(self, text, voice):
            captured_voices.append(voice)

        async def stream(self):
            yield {"type": "audio", "data": b"fake-mp3-bytes"}
            yield {"type": "WordBoundary"}  # non-audio events must be skipped

    fake_edge = SimpleNamespace(Communicate=FakeCommunicate)
    monkeypatch.setitem(sys.modules, "edge_tts", fake_edge)

    def fake_decode(data):
        assert data == b"fake-mp3-bytes"
        # 24000 Hz stereo, 0.5s -> after downmix+resample: ~8000 samples
        left = [1000] * 12000
        right = [-500] * 12000
        interleaved = [v for pair in zip(left, right) for v in pair]
        return SimpleNamespace(samples=interleaved, sample_rate=24000, nchannels=2)

    monkeypatch.setitem(sys.modules, "miniaudio", SimpleNamespace(decode=fake_decode))
    monkeypatch.setattr(tts.config, "TTS_PROVIDER", "edge")

    async def one_word(*words):
        for w in words:
            yield w

    async def consume():
        return [
            chunk
            async for chunk in tts.generate_audio_stream(one_word("سلام، آپ کیسے ہیں؟"))
        ]

    chunks = _run(consume())
    assert captured_voices == ["ur-PK-AsadNeural"]
    total = b"".join(chunks)
    # 24000 interleaved values -> 12000 mono -> /3 resample -> 4000 ulaw bytes
    assert len(total) == 4000
    assert all(isinstance(c, bytes) for c in chunks)


def test_english_sentence_routes_to_configured_provider(monkeypatch):
    """English stays on Deepgram; only non-Latin scripts go to Edge."""
    calls = {"edge": 0}

    async def fake_edge(sentence, lang="en"):
        calls["edge"] += 1
        yield b"x"

    async def fake_deepgram(sentence):
        yield b"y"

    monkeypatch.setattr(tts, "_speak_sentence_edge", fake_edge)
    monkeypatch.setattr(tts, "_speak_sentence_deepgram", fake_deepgram)
    monkeypatch.setattr(tts.config, "TTS_PROVIDER", "deepgram")

    async def one(*words):
        for w in words:
            yield w

    out = _run(_collect(tts.generate_audio_stream(one("Hello there. How are you?"))))
    assert b"".join(out) == b"yy"  # two sentences via deepgram
    assert calls["edge"] == 0


async def _collect(agen):
    return [c async for c in agen]
