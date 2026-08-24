import asyncio
import json
import logging

import websockets
from websockets.protocol import State

import config

logger = logging.getLogger(__name__)


class DeepgramSTT:
    def __init__(self, on_transcript_callback):
        self.url = (
            "wss://api.deepgram.com/v1/listen"
            "?encoding=mulaw&sample_rate=8000&channels=1"
            "&interim_results=true&endpointing=300&utterance_end_ms=1000"
        )
        self.on_transcript_callback = on_transcript_callback
        self.ws = None
        self.receive_task = None

    async def connect(self):
        extra_headers = {"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}
        self.ws = await websockets.connect(
            self.url,
            extra_headers=extra_headers,
            ping_interval=20,
            ping_timeout=10,
        )
        self.receive_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, audio_data: bytes):
        if self.ws and self.ws.state is State.OPEN:
            await self.ws.send(audio_data)

    async def _receive_loop(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "UtteranceEnd":
                    # No transcript content - nothing to do for now; endpointing
                    # already fires the callback with speech_final.
                    continue
                if msg_type == "Metadata":
                    continue

                if "channel" in data:
                    alternatives = data["channel"]["alternatives"]
                    if not alternatives:
                        continue
                    transcript = alternatives[0].get("transcript", "")
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    if transcript:
                        await self.on_transcript_callback(transcript, is_final or speech_final)
        except Exception as e:
            logger.warning("Deepgram STT receive loop ended: %s", e)

    async def close(self):
        if self.receive_task:
            self.receive_task.cancel()
        try:
            if self.ws and self.ws.state is State.OPEN:
                # Tell Deepgram we are done so it flushes final results
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
        except Exception:
            logger.debug("STT close ignored an error", exc_info=True)
