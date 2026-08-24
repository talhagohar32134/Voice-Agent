import asyncio
import base64
import csv
import io
import json
import logging
import secrets
import time

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
)
from fastapi.responses import FileResponse, RedirectResponse

import config
from call_queue import process_queue_batch
from database import CallLog, CallQueue, SessionLocal, Transcript
from llm import llm_manager, detect_mood, _EMPATHY_HINTS
from outbound import initiate_outbound_call, redirect_call_to_voicemail
from stt import DeepgramSTT
from telephony import get_inbound_twiml, get_outbound_twiml
from tts import generate_audio_stream

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("voice-agent")

app = FastAPI(title="Voice Agent")

# ------------------------------------------------------------------
# Turn latency metrics (in-memory, last N turns)
# ------------------------------------------------------------------

_METRICS_MAX_TURNS = 200
_metrics: list[dict] = []


def _record_turn_latency(stt_to_first_token_ms, first_token_to_first_audio_ms, total_ms):
    entry = {
        "ts": time.time(),
        "stt_to_first_token_ms": round(stt_to_first_token_ms),
        "first_token_to_first_audio_ms": round(first_token_to_first_audio_ms),
        "total_ms": round(total_ms),
    }
    _metrics.append(entry)
    if len(_metrics) > _METRICS_MAX_TURNS:
        del _metrics[: len(_metrics) - _METRICS_MAX_TURNS]


def _percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[idx]


@app.get("/stats")
async def stats():
    """Latency breakdown per conversational turn."""
    turns = list(_metrics)
    return {
        "turn_count": len(turns),
        "avg_total_ms": round(sum(t["total_ms"] for t in turns) / len(turns)) if turns else None,
        "avg_stt_to_first_token_ms": round(sum(t["stt_to_first_token_ms"] for t in turns) / len(turns)) if turns else None,
        "avg_first_token_to_first_audio_ms": (
            round(sum(t["first_token_to_first_audio_ms"] for t in turns) / len(turns)) if turns else None
        ),
        "p90_total_ms": _percentile([t["total_ms"] for t in turns], 0.9),
        "recent": turns[-10:],
    }


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_admin(x_api_key: str = Header(default=None)):
    """Admin endpoints trigger paid calls - lock them behind a key."""
    if not config.ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API disabled: ADMIN_API_KEY not set")
    if not x_api_key or not secrets.compare_digest(x_api_key, config.ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ------------------------------------------------------------------
# Sync DB helpers (run off the event loop via asyncio.to_thread)
# ------------------------------------------------------------------

def _persist_transcript_sync(call_id: str, role: str, text: str) -> int | None:
    db = SessionLocal()
    try:
        row = Transcript(call_id=call_id, role=role, text=text)
        db.add(row)
        db.commit()
        return row.id
    except Exception:
        logger.exception("Failed saving %s transcript", role)
        db.rollback()
        return None
    finally:
        db.close()


def _update_mood_sync(transcript_id: int, mood: str):
    db = SessionLocal()
    try:
        row = db.get(Transcript, transcript_id)
        if row:
            row.mood = mood
            db.commit()
    except Exception:
        logger.exception("Failed updating mood on transcript %s", transcript_id)
        db.rollback()
    finally:
        db.close()


def _update_call_log_status_sync(call_sid: str, status: str, duration: int | None = None):
    db = SessionLocal()
    try:
        log = db.query(CallLog).filter(CallLog.twilio_call_id == call_sid).first()
        if log:
            log.status = status
            if duration is not None:
                log.duration_seconds = duration
            db.commit()
    except Exception:
        logger.exception("Failed updating CallLog for %s", call_sid)
        db.rollback()
    finally:
        db.close()


# ------------------------------------------------------------------
# Twilio webhooks
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Land on the demo page directly."""
    return RedirectResponse(url="/demo")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/demo")
async def demo_page():
    """Browser demo: talk to the agent via laptop mic/speakers (no Twilio)."""
    return FileResponse("static/demo.html")


# ------------------------------------------------------------------
# Conversation dashboard (call transcripts + caller mood)
# ------------------------------------------------------------------

@app.get("/dashboard")
async def dashboard_page():
    return FileResponse("static/dashboard.html")


def _list_calls_sync():
    db = SessionLocal()
    try:
        from sqlalchemy import func

        rows = (
            db.query(
                Transcript.call_id,
                func.count(Transcript.id),
                func.min(Transcript.timestamp),
                func.max(Transcript.timestamp),
            )
            .group_by(Transcript.call_id)
            .order_by(func.max(Transcript.timestamp).desc())
            .limit(100)
            .all()
        )
        calls = []
        for call_id, msg_count, first_ts, last_ts in rows:
            moods = [
                r[0]
                for r in db.query(Transcript.mood)
                .filter(Transcript.call_id == call_id, Transcript.mood.isnot(None))
                .all()
            ]
            mood_summary = {}
            for m in moods:
                mood_summary[m] = mood_summary.get(m, 0) + 1
            dominant = (
                max(mood_summary.items(), key=lambda kv: kv[1])[0] if mood_summary else "unknown"
            )
            booked = (
                db.query(CallLog)
                .filter(CallLog.twilio_call_id == call_id)
                .count()
            )
            calls.append(
                {
                    "call_id": call_id,
                    "messages": msg_count,
                    "started": str(first_ts),
                    "last_activity": str(last_ts),
                    "moods": mood_summary,
                    "dominant_mood": dominant,
                }
            )
        return calls
    finally:
        db.close()


@app.get("/calls")
async def list_calls():
    return await asyncio.to_thread(_list_calls_sync)


def _get_transcripts_sync(call_id: str):
    db = SessionLocal()
    try:
        rows = (
            db.query(Transcript)
            .filter(Transcript.call_id == call_id)
            .order_by(Transcript.id)
            .all()
        )
        return [
            {
                "role": r.role,
                "text": r.text,
                "mood": r.mood,
                "timestamp": str(r.timestamp),
            }
            for r in rows
        ]
    finally:
        db.close()


@app.get("/calls/{call_id}/transcript")
async def call_transcript(call_id: str):
    return await asyncio.to_thread(_get_transcripts_sync, call_id)


@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    """Webhook for incoming Twilio calls."""
    host = request.headers.get("host")
    protocol = request.headers.get("x-forwarded-proto", "http")
    twiml = get_inbound_twiml(f"{protocol}://{host}")
    return Response(content=twiml, media_type="text/xml")


@app.post("/twilio/outbound")
async def twilio_outbound(request: Request):
    """Webhook hit when an outbound call is answered; connects media to our WS."""
    host = request.headers.get("host")
    protocol = request.headers.get("x-forwarded-proto", "http")
    twiml = get_outbound_twiml(f"{protocol}://{host}")
    return Response(content=twiml, media_type="text/xml")


_MACHINE_ANSWERS = {"machine_start", "machine_end_beep", "machine_end_silence", "fax"}


@app.post("/twilio/amd_status")
async def twilio_amd_status(request: Request):
    """Callback for Answering Machine Detection - divert machines to voicemail."""
    form = await request.form()
    answered_by = form.get("AnsweredBy")
    call_sid = form.get("CallSid")
    logger.info("AMD status for %s: %s", call_sid, answered_by)

    if call_sid and answered_by in _MACHINE_ANSWERS:
        await asyncio.to_thread(redirect_call_to_voicemail, call_sid)
        await asyncio.to_thread(_update_call_log_status_sync, call_sid, "voicemail")

    return Response(status_code=200)


@app.post("/twilio/status")
async def twilio_status(request: Request):
    """Final call status updates from Twilio (completed/busy/failed/no-answer)."""
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus", "unknown")
    try:
        duration = int(form.get("CallDuration", 0))
    except (TypeError, ValueError):
        duration = None

    logger.info("Call %s finished: %s (%ss)", call_sid, call_status, duration)
    if call_sid:
        await asyncio.to_thread(_update_call_log_status_sync, call_sid, call_status, duration)
        llm_manager.end_session(call_sid)

    return Response(status_code=200)


# ------------------------------------------------------------------
# Admin endpoints (auth required)
# ------------------------------------------------------------------

@app.post("/call", dependencies=[Depends(require_admin)])
async def trigger_single_call(phone_number: str = Form(...), db=Depends(get_db)):
    """Manually trigger a single outbound call."""
    call_sid = await asyncio.to_thread(initiate_outbound_call, phone_number)
    db.add(CallLog(twilio_call_id=call_sid, direction="outbound", status="queued", phone_number=phone_number))
    db.commit()
    return {"status": "ok", "call_sid": call_sid}


@app.post("/call/batch", dependencies=[Depends(require_admin)])
async def trigger_batch_calls(background_tasks: BackgroundTasks):
    """Trigger processing of the call queue."""
    background_tasks.add_task(process_queue_batch)
    return {"status": "batch processing started"}


@app.post("/call/csv", dependencies=[Depends(require_admin)])
async def upload_csv_queue(file: UploadFile = File(...), db=Depends(get_db)):
    """Upload a CSV with 'phone_number' and 'context' headers to the queue."""
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))

    count = 0
    for row in reader:
        phone = (row.get("phone_number") or "").strip()
        context = row.get("context", "")
        if phone:
            db.add(CallQueue(phone_number=phone, context=context))
            count += 1

    db.commit()
    return {"status": "ok", "queued": count}


# ------------------------------------------------------------------
# Media stream websocket
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    stream_sid = None
    call_sid = None

    # Per-call state
    agent_speaking = False
    tts_task = None
    last_response_started = 0.0
    caller_moods: list[str] = []  # rolling mood history for empathy
    last_mood_at = 0.0

    async def send_media(audio_chunk: bytes):
        payload = base64.b64encode(audio_chunk).decode("ascii")
        msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": payload}}
        await websocket.send_text(json.dumps(msg))

    async def finish_assistant_turn(spoken_text: str):
        """Persist whatever the caller heard (complete or interrupted response)."""
        text = spoken_text.strip()
        if text:
            llm_manager.add_assistant_message(call_sid, text)
            await asyncio.to_thread(_persist_transcript_sync, call_sid, "agent", text)

    async def speak_text(text: str):
        """Stream one fixed string (e.g. outbound greeting) through TTS."""
        nonlocal agent_speaking

        spoken_text = text  # per-response local state - no cross-turn races

        async def _gen():
            yield text

        agent_speaking = True
        try:
            async for chunk in generate_audio_stream(_gen()):
                await send_media(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            agent_speaking = False
            await finish_assistant_turn(spoken_text)

    async def run_agent_response(turn_started_at: float | None = None):
        nonlocal agent_speaking
        agent_speaking = True

        spoken_text = ""  # per-response local state - no cross-turn races
        first_token_at = None

        async def tee_stream(token_stream):
            nonlocal spoken_text, first_token_at
            async for token in token_stream:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                spoken_text += token
                yield token

        try:
            # Empathy: if the caller's recent mood is negative, prime the agent
            extra = ""
            recent = caller_moods[-3:]
            if recent:
                dominant = max(set(recent), key=recent.count)
                hint = _EMPATHY_HINTS.get(dominant)
                if hint and dominant != "positive":
                    extra = hint

            audio_stream = generate_audio_stream(
                tee_stream(llm_manager.generate_response(call_sid, extra_system=extra))
            )
            async for chunk in audio_stream:
                if not agent_speaking:
                    break
                await send_media(chunk)
                if turn_started_at is not None and first_token_at is not None:
                    # First audio went out on the wire - record the full breakdown once.
                    now = time.monotonic()
                    _record_turn_latency(
                        (first_token_at - turn_started_at) * 1000,
                        (now - first_token_at) * 1000,
                        (now - turn_started_at) * 1000,
                    )
                    logger.info(
                        "Turn latency: total=%.0fms llm_first_token=%.0fms tts_first_audio=%.0fms",
                        (now - turn_started_at) * 1000,
                        (first_token_at - turn_started_at) * 1000,
                        (now - first_token_at) * 1000,
                    )
                    turn_started_at = None  # only measure the first chunk
        except asyncio.CancelledError:
            pass
        finally:
            agent_speaking = False
            await finish_assistant_turn(spoken_text)

    async def on_transcript(text: str, is_final: bool):
        nonlocal agent_speaking, tts_task, last_response_started, caller_moods

        text = (text or "").strip()
        if not text:
            return

        # Echo guard: only *final* transcripts of meaningful length may barge in,
        # so the agent's own TTS bleeding into the mic doesn't self-interrupt.
        if agent_speaking:
            if not is_final or len(text) < config.MIN_INTERRUPT_CHARS:
                return
            logger.info("Interruption detected: %r", text)
            if tts_task and not tts_task.done():
                tts_task.cancel()
            if stream_sid:
                try:
                    await websocket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
                except Exception:
                    logger.exception("Failed sending clear event")
            agent_speaking = False

        # Debounce: ignore near-duplicate finals fired right after a response began
        if time.monotonic() - last_response_started < 0.25:
            return

        logger.info("Caller: %s", text)
        llm_manager.add_user_message(call_sid, text)
        row_id = await asyncio.to_thread(_persist_transcript_sync, call_sid, "user", text)

        async def _mood_worker():
            # Throttle: max one mood LLM call per 10s per call (free-tier budget).
            nonlocal last_mood_at
            now_m = time.monotonic()
            if now_m - last_mood_at < 10:
                mood = caller_moods[-1] if caller_moods else "neutral"
            else:
                last_mood_at = now_m
                mood = await detect_mood(text)
            logger.info("Caller mood: %s", mood)
            caller_moods.append(mood)
            if row_id:
                await asyncio.to_thread(_update_mood_sync, row_id, mood)

        asyncio.create_task(_mood_worker())

        last_response_started = time.monotonic()
        turn_started_at = last_response_started
        if tts_task and not tts_task.done():
            tts_task.cancel()
        tts_task = asyncio.create_task(run_agent_response(turn_started_at))

    try:
        stt = DeepgramSTT(on_transcript)
        await stt.connect()
    except Exception:
        logger.exception("Could not connect to STT provider - closing stream")
        await websocket.close()
        return

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")

            if event == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                custom_params = data["start"].get("customParameters") or {}
                is_outbound = str(custom_params.get("is_outbound", "")).lower() == "true"

                llm_manager.ensure_session(call_sid, is_outbound=is_outbound)
                logger.info("%s call %s connected.", "Outbound" if is_outbound else "Inbound", call_sid)

                if is_outbound:
                    tts_task = asyncio.create_task(speak_text(config.OUTBOUND_GREETING))

            elif event == "media":
                audio_bytes = base64.b64decode(data["media"]["payload"])
                await stt.send_audio(audio_bytes)

            elif event == "stop":
                logger.info("Call %s stream stopped", call_sid)
                break

    except Exception:
        logger.exception("WebSocket error on call %s", call_sid)
    finally:
        if tts_task and not tts_task.done():
            tts_task.cancel()
        await stt.close()
