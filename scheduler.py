"""Appointment scheduling - the single source of truth for bookings.

The LLM can only touch bookings through these functions via tool-calling,
so it can never claim a booking that does not exist in the database.
"""

import logging
from datetime import date, datetime, time, timedelta

from database import Appointment, SessionLocal

import config

logger = logging.getLogger(__name__)

SLOT_MINUTES = 30


def _clinic_windows() -> list[tuple[time, time]]:
    """Bookable windows per weekday (Mon=0 .. Sun=6)."""
    start = time(config.CLINIC_HOUR_START, 0)
    end = time(config.CLINIC_HOUR_END, 0)
    # Mon-Fri same window; weekend closed
    return [(start, end)] * 5 + [(start, start), (start, start)]


def _slots_for_date(d: date) -> list[datetime]:
    windows = _clinic_windows()
    open_start, close_end = windows[d.weekday()]
    if open_start == close_end:
        return []  # closed that day
    slots = []
    cur = datetime.combine(d, open_start)
    last = datetime.combine(d, close_end)
    while cur + timedelta(minutes=SLOT_MINUTES) <= last:
        slots.append(cur)
        cur += timedelta(minutes=SLOT_MINUTES)
    return slots


def check_availability(date_str: str) -> str:
    """Return free slots for YYYY-MM-DD."""
    try:
        d = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return f"ERROR: Could not parse date {date_str!r}. Use YYYY-MM-DD."

    if d < date.today():
        return "ERROR: That date is in the past."

    booked = _booked_times(d)
    free = [s for s in _slots_for_date(d) if s not in booked]

    if not free:
        return f"No open slots on {date_str}. Clinic hours: {config.CLINIC_HOUR_START}:00-{config.CLINIC_HOUR_END}:00, Mon-Fri."
    return "Open slots on " + date_str + ": " + ", ".join(s.strftime("%H:%M") for s in free)


def book_appointment(date_str: str, time_str: str, phone_number: str = "", patient_name: str = "", call_id: str = "") -> str:
    """Book a slot. Returns success text or an actionable error for the LLM."""
    try:
        d = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return f"ERROR: Bad date {date_str!r} (need YYYY-MM-DD)."

    try:
        t = datetime.strptime(time_str.strip(), "%H:%M").time()
    except ValueError:
        return f"ERROR: Bad time {time_str!r} (need HH:MM, 24h)."

    slot = datetime.combine(d, t)
    if slot not in _slots_for_date(d):
        return "ERROR: That time is outside clinic hours. Use check_availability first."
    if slot < datetime.now():
        return "ERROR: Cannot book in the past."
    if slot in _booked_times(d):
        return "ERROR: Slot already taken. Offer another slot from check_availability."

    db = SessionLocal()
    try:
        db.add(Appointment(
            phone_number=phone_number,
            patient_name=patient_name[:120],
            appointment_time=slot,
            call_id=call_id or "",
        ))
        db.commit()
        logger.info("BOOKED %s %s for %s (call %s)", date_str, time_str, phone_number or patient_name, call_id)
        return (
            f"SUCCESS: Booked {slot.strftime('%A, %d %B %Y at %H:%M')}. "
            "Tell the caller to arrive 10 minutes early with their ID."
        )
    except Exception:
        logger.exception("Booking failed")
        db.rollback()
        return "ERROR: Database failure while booking. Ask the caller to try again later."
    finally:
        db.close()


def list_appointments(phone_number: str = "") -> str:
    """List upcoming bookings, optionally filtered by caller number."""
    db = SessionLocal()
    try:
        q = db.query(Appointment).filter(
            Appointment.status == "booked",
            Appointment.appointment_time >= datetime.now(),
        )
        if phone_number:
            q = q.filter(Appointment.phone_number == phone_number)
        rows = q.order_by(Appointment.appointment_time).limit(10).all()

        if not rows:
            return "No upcoming appointments found."
        lines = [
            f"{r.appointment_time.strftime('%a %d %b %H:%M')} - {r.patient_name or r.phone_number}"
            for r in rows
        ]
        return "Upcoming appointments: " + "; ".join(lines)
    finally:
        db.close()


def _booked_times(d: date) -> set[datetime]:
    day_start = datetime.combine(d, time(0, 0))
    day_end = day_start + timedelta(days=1)
    db = SessionLocal()
    try:
        rows = (
            db.query(Appointment.appointment_time)
            .filter(
                Appointment.status == "booked",
                Appointment.appointment_time >= day_start,
                Appointment.appointment_time < day_end,
            )
            .all()
        )
        return {r[0] for r in rows}
    finally:
        db.close()


# ------------------------------------------------------------------
# Tool schema exposed to the LLM (OpenAI/Groq function-calling format)
# ------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "List open appointment slots for a given date. Always call this before offering times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date to check, format YYYY-MM-DD"}
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment slot after confirming date and time with the caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Appointment date, YYYY-MM-DD"},
                    "time": {"type": "string", "description": "Appointment start time, HH:MM 24h"},
                    "patient_name": {"type": "string", "description": "Caller's name if known, else empty"},
                },
                "required": ["date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_appointments",
            "description": "List the caller's upcoming appointments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(name: str, arguments_json: str, context: dict) -> str:
    """Run a tool by name with raw JSON args. Returns plain-text result for the LLM."""
    import json

    try:
        args = json.loads(arguments_json) if arguments_json and arguments_json.strip() else {}
    except json.JSONDecodeError:
        return f"ERROR: Invalid JSON arguments: {arguments_json!r}"

    if name == "check_availability":
        return check_availability(args.get("date", ""))
    if name == "book_appointment":
        return book_appointment(
            date_str=args.get("date", ""),
            time_str=args.get("time", ""),
            patient_name=args.get("patient_name", "") or "",
            phone_number=context.get("phone_number", ""),
            call_id=context.get("call_id", ""),
        )
    if name == "list_appointments":
        return list_appointments(phone_number=context.get("phone_number", ""))
    return f"ERROR: Unknown tool {name}"
