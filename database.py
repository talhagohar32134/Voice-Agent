from sqlalchemy import create_engine, event, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import logging

import config

logger = logging.getLogger(__name__)

_DB_FILE = config.DB_FILE


def _build_engine():
    """
    Prefer SQLCipher (encryption at rest - transcripts contain PHI).
    Fall back to plain sqlite3 on platforms without a driver (e.g. Windows),
    with a loud warning so production never silently runs unencrypted.
    """
    password = config.DB_PASSWORD
    if not password or password == "default_secret":
        logger.warning(
            "DB_PASSWORD not set - running with UNENCRYPTED sqlite3. "
            "Do NOT use this in production with real caller data."
        )
        return create_engine(f"sqlite:///{_DB_FILE}", echo=False)

    try:
        import sqlcipher3  # noqa: F401  (Linux/macOS wheels via sqlcipher3-binary)
    except ImportError:
        logger.warning(
            "sqlcipher3 driver not installed (unavailable on Windows). "
            "Falling back to UNENCRYPTED sqlite3. Do NOT use this in production."
        )
        return create_engine(f"sqlite:///{_DB_FILE}", echo=False)

    def _connect():
        conn = sqlcipher3.connect(_DB_FILE)
        conn.execute(f"PRAGMA key = '{password}'")
        return conn

    engine = create_engine("sqlite://", creator=_connect, echo=False)
    logger.info("Using SQLCipher (encrypted) database.")
    return engine


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(Integer, primary_key=True, index=True)
    twilio_call_id = Column(String, index=True, unique=True)
    direction = Column(String)  # "inbound" or "outbound"
    status = Column(String)  # "queued", "in-progress", "completed", "no-answer", "voicemail", "failed"
    phone_number = Column(String, index=True)
    duration_seconds = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Transcript(Base):
    __tablename__ = "transcripts"
    # WARNING: This table stores PHI (Protected Health Information) from caller speech.
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String, index=True)  # twilio_call_id
    role = Column(String)  # "user" or "agent"
    text = Column(String)
    mood = Column(String)  # caller mood at this utterance (user rows only)
    timestamp = Column(DateTime, default=datetime.utcnow)


def _auto_migrate():
    """Lightweight sqlite migrations - add missing columns if needed."""
    import sqlalchemy

    additions = {
        "transcripts": ["ALTER TABLE transcripts ADD COLUMN mood TEXT"],
    }
    with engine.connect() as conn:
        for _table, stmts in additions.items():
            for stmt in stmts:
                try:
                    conn.execute(sqlalchemy.text(stmt))
                    conn.commit()
                    logger.info("Migration applied: %s", stmt)
                except Exception:
                    pass  # column already exists


class CallQueue(Base):
    __tablename__ = "call_queue"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, index=True)
    context = Column(String)  # JSON or string containing appointment details, etc.
    status = Column(String, default="pending")  # "pending", "processing", "completed", "failed"
    timestamp = Column(DateTime, default=datetime.utcnow)


class DoNotCall(Base):
    __tablename__ = "dnc_list"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)


class ConsentRecord(Base):
    __tablename__ = "consent_records"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, unique=True, index=True)
    opt_in_timestamp = Column(DateTime, default=datetime.utcnow)


class Appointment(Base):
    """Real bookings created via LLM tool-calling - source of truth."""
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String, index=True)
    patient_name = Column(String, default="")
    appointment_time = Column(DateTime, index=True)  # clinic-local naive datetime
    status = Column(String, default="booked")  # booked / cancelled / completed
    call_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)
_auto_migrate()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
