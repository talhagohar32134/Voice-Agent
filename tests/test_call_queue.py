import asyncio
from unittest.mock import patch

import call_queue
from database import CallLog, CallQueue, DoNotCall, ConsentRecord


def test_dnc_number_is_blocked(db):
    db.add(DoNotCall(phone_number="+15551110000"))
    db.add(ConsentRecord(phone_number="+15551110000"))
    db.commit()

    assert call_queue.can_call_number(db, "+15551110000") is False


def test_missing_consent_is_blocked(db):
    assert call_queue.can_call_number(db, "+15552220000") is False


def test_consent_record_without_timestamp_is_blocked(db):
    db.add(ConsentRecord(phone_number="+15553330000", opt_in_timestamp=None))
    db.commit()
    assert call_queue.can_call_number(db, "+15553330000") is False


def _with_hours(monkeypatch, allowed: bool):
    monkeypatch.setattr(call_queue, "is_within_calling_hours", lambda n: allowed)


def test_valid_number_passes_all_checks(db, consented_number, monkeypatch):
    _with_hours(monkeypatch, True)
    assert call_queue.can_call_number(db, consented_number) is True


def test_outside_hours_is_blocked(db, consented_number, monkeypatch):
    _with_hours(monkeypatch, False)
    assert call_queue.can_call_number(db, consented_number) is False


def test_batch_dials_only_policy_compliant_numbers(db, monkeypatch):
    ok_number = "+15551234567"      # has consent (fixture adds it below)
    dnc_number = "+15559990000"     # on DNC list

    from datetime import datetime

    db.add(ConsentRecord(phone_number=ok_number, opt_in_timestamp=datetime.utcnow()))
    db.add(ConsentRecord(phone_number=dnc_number, opt_in_timestamp=datetime.utcnow()))
    db.add(DoNotCall(phone_number=dnc_number))
    no_consent = "+15557770000"

    for num in (ok_number, dnc_number, no_consent):
        db.add(CallQueue(phone_number=num))
    db.commit()

    dialed = []

    def fake_dial(number):
        dialed.append(number)
        return f"CA-{number}"

    monkeypatch.setattr(call_queue, "initiate_outbound_call", fake_dial)
    _with_hours(monkeypatch, True)

    asyncio.run(call_queue.process_queue_batch())

    statuses = {q.phone_number: q.status for q in db.query(CallQueue).all()}
    assert statuses[ok_number] == "completed"
    assert statuses[dnc_number] == "skipped_policy"
    assert statuses[no_consent] == "skipped_policy"
    assert dialed == [ok_number]

    log = db.query(CallLog).filter_by(phone_number=ok_number).first()
    assert log is not None
    assert log.twilio_call_id == f"CA-{ok_number}"


def test_batch_marks_failed_when_dialing_raises(db, consented_number, monkeypatch):
    db.add(CallQueue(phone_number=consented_number))
    db.commit()

    def boom(number):
        raise RuntimeError("twilio down")

    monkeypatch.setattr(call_queue, "initiate_outbound_call", boom)
    _with_hours(monkeypatch, True)

    asyncio.run(call_queue.process_queue_batch())

    item = db.query(CallQueue).filter_by(phone_number=consented_number).first()
    assert item.status == "failed"


def test_unknown_timezone_denied():
    with patch.object(call_queue.timezone, "time_zones_for_number", return_value=("Etc/Unknown",)):
        assert call_queue.is_within_calling_hours("+15550000000") is False
