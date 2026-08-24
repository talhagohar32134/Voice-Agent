import secrets

from dotenv import load_dotenv

load_dotenv()

import os  # noqa: E402  (must read env after load_dotenv)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# "multi" lets Deepgram auto-detect English/Hindi/Urdu per utterance
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "multi")

# --- LLM provider -----------------------------------------------------------
# "anthropic" (Claude) or "groq" (free tier, OpenAI-compatible, Llama models)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# --- TTS provider ------------------------------------------------------------
# "elevenlabs" or "deepgram" (Aura - billed from the same $200 Deepgram credit)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
DEEPGRAM_TTS_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")

# --- Edge TTS (free, no key) - used for non-English sentences automatically ---
EDGE_VOICE_EN = os.getenv("EDGE_VOICE_EN", "en-US-AvaNeural")
EDGE_VOICE_UR = os.getenv("EDGE_VOICE_UR", "ur-PK-AsadNeural")
EDGE_VOICE_HI = os.getenv("EDGE_VOICE_HI", "hi-IN-SwaraNeural")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Keep your responses brief.")
OUTBOUND_GREETING = os.getenv("OUTBOUND_GREETING", "Hello, how can I help you today?")
VOICEMAIL_MESSAGE = os.getenv(
    "VOICEMAIL_MESSAGE",
    "Hi, this is a message from the healthcare clinic. Please call us back at your earliest convenience. Thank you!",
)

DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_FILE = os.getenv("DB_FILE", "voice_agent.db")

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Admin endpoints (/call, /call/batch, /call/csv) are dangerous - they trigger
# paid outbound calls. If ADMIN_API_KEY is unset they stay disabled entirely.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Calling-hours window for outbound (local time of the callee), TCPA guidance
CALLING_HOUR_START = int(os.getenv("CALLING_HOUR_START", "9"))
CALLING_HOUR_END = int(os.getenv("CALLING_HOUR_END", "18"))

# Clinic bookable window (Mon-Fri) used by the appointment tools
CLINIC_HOUR_START = int(os.getenv("CLINIC_HOUR_START", "9"))
CLINIC_HOUR_END = int(os.getenv("CLINIC_HOUR_END", "17"))

# Interruption guard: ignore short barge-in blips (echo of agent's own TTS,
# coughs, noise). Final transcripts shorter than this never cancel the agent.
MIN_INTERRUPT_CHARS = int(os.getenv("MIN_INTERRUPT_CHARS", "2"))


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)
