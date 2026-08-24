# Voice Agent — AI Phone Receptionist

Production-style voice agent for a healthcare clinic: answers calls, talks
naturally (English / Urdu / Hindi with live code-switching), books **real**
appointments, reads the caller's mood, and logs everything to a dashboard.

Built on Twilio media streams + Deepgram STT + Groq LLM + Edge/Deepgram TTS.
Runs fully on free tiers.

---

## Architecture

```
Caller (phone/browser mic)
        │
   Twilio Media Stream (or browser WS demo)
        │  mulaw 8kHz over WebSocket
        ▼
  FastAPI server (main.py)
        │
        ├── Deepgram STT (nova-3 multi) ──► transcripts, auto language detect
        ├── Mood classifier ────────────► positive/neutr../angry (+ empathy hints)
        ├── Groq LLM (gpt-oss-20b) ─────► replies + FUNCTION CALLING:
        │       check_availability / book_appointment / list_appointments
        │       (real DB writes - the agent cannot fake bookings)
        ├── TTS router ─────────────────► English: Deepgram Aura
        │                                Urdu/Hindi: Edge neural voices (free)
        └── SQLite (SQLCipher-ready) ───► call logs, transcripts+moods,
                                         appointments, DNC list, consent records
```

## Features

- **Real-time voice conversation** with barge-in (caller can interrupt anytime)
- **Trilingual**: English ↔ Urdu ↔ Hindi mid-sentence switching; Roman Urdu supported
- **Real appointment booking** via function calling — availability checked against DB,
  double-booking impossible, agent instructed to never claim a booking that didn't happen
- **Mood detection** per utterance (keyword fast-path + LLM fallback); negative moods
  prime the next reply with empathy instructions
- **Live latency metrics** (`/stats`): STT→LLM→TTS breakdown per turn
- **Conversation dashboard** (`/dashboard`): all calls, full transcripts, mood badges,
  auto-refreshes every 4s during live conversations
- **TCPA compliance guards** for outbound: DNC list, consent records, calling-hours window
- **Admin auth** on call-triggering endpoints (`X-API-Key` header)
- **Browser demo mode** (`/demo`) — test the whole pipeline from laptop mic, no Twilio needed

## Quick Start

```bash
pip install -r requirements.txt
copy .env.example .env      # then fill in keys below
uvicorn main:app --port 8000
```

### Required API keys (all free, no credit card)

| Key | Where | Cost |
|-----|-------|------|
| `DEEPGRAM_API_KEY` | console.deepgram.com | $200 free credit |
| `GROQ_API_KEY` | console.groq.com | free forever |
| `TWILIO_*` | console.twilio.com | trial credit (only needed for real phone calls) |

Optional: `ANTHROPIC_API_KEY` + `LLM_PROVIDER=anthropic`, or ElevenLabs TTS.

Then open **http://localhost:8000/demo**, click *Start Talking*, and speak.

### Real phone calls (Twilio)

1. Put a public URL in `.env` → `BASE_URL` (e.g. `ngrok http 8000`)
2. Twilio phone number → Voice webhook → `<BASE_URL>/twilio/inbound` (HTTP POST)
3. Trigger outbound: `POST /call` with header `X-API-Key: <ADMIN_API_KEY>` and
   form field `phone_number`

## Endpoints

| Route | What |
|-------|------|
| `GET /demo` | Browser mic demo page |
| `GET /dashboard` | Conversation + mood dashboard |
| `GET /calls`, `GET /calls/{id}/transcript` | Dashboard JSON APIs |
| `GET /stats` | Per-turn latency breakdown |
| `GET /health` | Health check |
| `POST /twilio/inbound` `/twilio/outbound` `/twilio/amd_status` `/twilio/status` | Twilio webhooks |
| `POST /call` `/call/batch` `/call/csv` | Outbound dialing (admin key required) |

## Testing

```bash
python -m pytest tests -q
```

60 tests cover TwiML generation, queue policy logic, webhooks/auth, LLM tool-calling
round-trips, scheduler rules, language detection, TTS pipeline, mood classification
and dashboard APIs — all offline/mocked, no API spend.

## Project Structure

```
main.py          FastAPI app: webhooks, WS media loop, mood wiring, dashboard APIs
llm.py           Groq/Anthropic streaming + function-calling + mood classifier
tts.py           Sentence-chunked TTS router (Deepgram Aura / Edge / ElevenLabs)
stt.py           Deepgram streaming STT client
scheduler.py     Appointment tools (single source of truth for bookings)
call_queue.py    Outbound batch dialer with TCPA checks
telephony.py     TwiML builders
database.py      Models + SQLCipher-or-sqlite engine + light migrations
outbound.py      Twilio REST: dialing, AMD voicemail redirect
static/          demo.html, dashboard.html
tests/           60 pytest tests
```

## Known Trade-offs

- Free Groq tier is rate-limited (~30 req/min) - bursts can add a few seconds of delay;
  mood detection is throttled to protect the budget
- Deepgram multi mode transcribes Urdu speech as Hindi script (meaning preserved);
  agent still replies in Urdu/Roman Urdu as instructed
- Windows dev runs unencrypted SQLite (SQLCipher wheels unavailable); production Linux
  gets encryption at rest automatically

## Roadmap

- Provider fallbacks (Groq→Anthropic) + mid-call STT reconnect
- SMS confirmations after booking
- Multi-campaign support, load testing, CI
