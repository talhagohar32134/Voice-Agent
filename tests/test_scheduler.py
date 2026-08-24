from datetime import date, datetime, time, timedelta

import scheduler
from database import Appointment


def _next_weekday(days_ahead=1) -> str:
    """Next Mon-Fri date from tomorrow onwards, as YYYY-MM-DD."""
    d = date.today() + timedelta(days=days_ahead)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def test_availability_lists_clinic_slots():
    result = scheduler.check_availability(_next_weekday())
    assert result.startswith("Open slots on")
    assert "09:00" in result
    assert "16:30" in result  # last slot ending at 17:00


def test_availability_rejects_past_date():
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    assert "ERROR" in scheduler.check_availability(yesterday)


def test_availability_rejects_bad_format():
    assert "ERROR" in scheduler.check_availability("tomorrow")


def test_weekend_is_closed():
    # Find next Sunday
    d = date.today() + timedelta(days=1)
    while d.weekday() != 6:
        d += timedelta(days=1)
    result = scheduler.check_availability(d.strftime("%Y-%m-%d"))
    assert result.startswith("No open slots")


def test_booking_creates_row(db):
    day = _next_weekday()
    result = scheduler.book_appointment(day, "10:00", patient_name="Talha", phone_number="+15551234567")
    assert "SUCCESS" in result

    row = db.query(Appointment).filter_by(patient_name="Talha").first()
    assert row is not None
    expected = datetime.combine(date.fromisoformat(day), time(10, 0))
    assert row.appointment_time == expected
    assert row.status == "booked"


def test_double_booking_rejected(db):
    day = _next_weekday()
    assert "SUCCESS" in scheduler.book_appointment(day, "10:00")
    second = scheduler.book_appointment(day, "10:00", patient_name="Other")
    assert "ERROR" in second

    count = db.query(Appointment).filter_by(appointment_time=datetime.combine(
        date.fromisoformat(day), time(10, 0)
    )).count()
    assert count == 1


def test_booking_outside_hours_rejected(db):
    day = _next_weekday()
    early = scheduler.book_appointment(day, "07:30")   # before clinic opens
    late = scheduler.book_appointment(day, "23:00")    # after close
    assert "ERROR" in early
    assert "ERROR" in late
    assert db.query(Appointment).count() == 0


def test_booked_slot_hidden_from_availability(db):
    day = _next_weekday()
    scheduler.book_appointment(day, "10:00")
    avail = scheduler.check_availability(day)
    assert "10:00" not in avail
    assert "09:00" in avail  # others still open


def test_execute_tool_dispatch(db):
    day = _next_weekday()
    import json

    args = json.dumps({"date": day, "time": "11:00"})
    result = scheduler.execute_tool("book_appointment", args, {"phone_number": "+15550001", "call_id": "C1"})
    assert "SUCCESS" in result

    unknown = scheduler.execute_tool("nope", "{}", {})
    assert "Unknown tool" in unknown

    bad_json = scheduler.execute_tool("check_availability", "{not json", {})
    assert "Invalid JSON" in bad_json
