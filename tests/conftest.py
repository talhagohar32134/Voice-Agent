"""Shared test setup.

Environment variables MUST be configured before any project module is imported,
because config.py snapshots them at import time. load_dotenv() does not override
already-set os.environ keys, so what we set here always wins over .env.
"""

import os
import tempfile
from datetime import datetime

_TEST_DB = os.path.join(tempfile.mkdtemp(prefix="va_test_"), "test_voice_agent.db")

os.environ["DB_FILE"] = _TEST_DB
os.environ["DB_PASSWORD"] = ""  # plain sqlite in tests
os.environ["ADMIN_API_KEY"] = "test-admin-key"
os.environ["TWILIO_ACCOUNT_SID"] = "AC-test-sid"
os.environ["TWILIO_AUTH_TOKEN"] = "test-auth-token"
os.environ["TWILIO_PHONE_NUMBER"] = "+15550000000"
os.environ["DEEPGRAM_API_KEY"] = "test-deepgram"
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic"
os.environ["ELEVENLABS_API_KEY"] = "test-elevenlabs"
os.environ["ELEVENLABS_VOICE_ID"] = "test-voice"
# Mood-detection tests monkeypatch the Groq client - force the provider so
# the suite is hermetic even when a developer .env says something else
# (and on CI, where no .env exists at all).
os.environ["LLM_PROVIDER"] = "groq"
os.environ["GROQ_API_KEY"] = "test-groq"

import pytest  # noqa: E402

from database import Base, ConsentRecord, SessionLocal, engine  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    """Fresh tables for every test."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()


@pytest.fixture
def db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def consented_number(db):
    """A phone number that has opted in - passes the consent check."""
    num = "+15551234567"
    db.add(ConsentRecord(phone_number=num, opt_in_timestamp=datetime.utcnow()))
    db.commit()
    return num


@pytest.fixture
def api_client(monkeypatch):
    """TestClient with external services stubbed out."""
    from fastapi.testclient import TestClient

    import main

    # Never hit the real Twilio REST API from tests
    monkeypatch.setattr(
        main, "initiate_outbound_call", lambda n: f"CA-fake-{abs(hash(n)) % 10**8}"
    )
    monkeypatch.setattr(main, "redirect_call_to_voicemail", lambda sid: None)

    with TestClient(main.app) as client:
        yield client


AUTH_HEADERS = {"X-API-Key": "test-admin-key"}
