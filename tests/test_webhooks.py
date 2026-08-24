from database import CallLog


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_inbound_webhook_returns_stream_twiml(api_client):
    resp = api_client.post("/twilio/inbound")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/xml")
    # TestClient host is "testserver" and proto defaults to http -> ws://
    assert "ws://testserver/ws" in resp.text
    assert "<Connect>" in resp.text


def test_outbound_webhook_includes_custom_param(api_client):
    resp = api_client.post("/twilio/outbound")
    assert resp.status_code == 200
    assert 'name="is_outbound"' in resp.text


# ------------------------------------------------------------------
# Admin auth
# ------------------------------------------------------------------

def test_admin_endpoints_reject_missing_key(api_client):
    resp = api_client.post("/call", data={"phone_number": "+15551234567"})
    assert resp.status_code == 401


def test_admin_endpoints_reject_wrong_key(api_client):
    resp = api_client.post(
        "/call", data={"phone_number": "+15551234567"}, headers={"X-API-Key": "nope"}
    )
    assert resp.status_code == 401


def test_admin_disabled_entirely_when_key_unset(api_client, monkeypatch):
    import config

    monkeypatch.setattr(config, "ADMIN_API_KEY", None)
    resp = api_client.post(
        "/call", data={"phone_number": "+15551234567"}, headers={"X-API-Key": "test-admin-key"}
    )
    assert resp.status_code == 503


# ------------------------------------------------------------------
# Call triggering + CSV queue
# ------------------------------------------------------------------

def test_single_call_creates_log(api_client, db):
    resp = api_client.post(
        "/call",
        data={"phone_number": "+15551234567"},
        headers={"X-API-Key": "test-admin-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["call_sid"].startswith("CA-fake-")

    log = db.query(CallLog).filter_by(twilio_call_id=body["call_sid"]).first()
    assert log is not None
    assert log.direction == "outbound"
    assert log.status == "queued"


def test_csv_upload_queues_rows(api_client, db):
    csv_content = (
        "phone_number,context\n"
        "+15551110001,reminder A\n"
        "+15551110002,reminder B\n"
        ",missing number\n"
    )
    resp = api_client.post(
        "/call/csv",
        files={"file": ("calls.csv", csv_content.encode(), "text/csv")},
        headers={"X-API-Key": "test-admin-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["queued"] == 2


# ------------------------------------------------------------------
# Twilio status / AMD webhooks
# ------------------------------------------------------------------

def _seed_call(db, sid):
    log = CallLog(twilio_call_id=sid, direction="outbound", status="queued", phone_number="+15550001111")
    db.add(log)
    db.commit()
    return sid


def test_status_callback_updates_call_log(api_client, db):
    sid = _seed_call(db, "CA-status-test")

    resp = api_client.post(
        "/twilio/status",
        data={"CallSid": sid, "CallStatus": "completed", "CallDuration": "42"},
    )
    assert resp.status_code == 200

    log = db.query(CallLog).filter_by(twilio_call_id=sid).first()
    assert log.status == "completed"
    assert log.duration_seconds == 42


def test_amd_machine_redirects_to_voicemail(api_client, db, monkeypatch):
    import main

    redirected = []
    monkeypatch.setattr(main, "redirect_call_to_voicemail", lambda sid: redirected.append(sid))

    sid = _seed_call(db, "CA-amd-test")

    resp = api_client.post(
        "/twilio/amd_status",
        data={"CallSid": sid, "AnsweredBy": "machine_end_beep"},
    )
    assert resp.status_code == 200
    assert redirected == [sid]

    log = db.query(CallLog).filter_by(twilio_call_id=sid).first()
    assert log.status == "voicemail"


def test_amd_human_does_not_redirect(api_client, db, monkeypatch):
    import main

    redirected = []
    monkeypatch.setattr(main, "redirect_call_to_voicemail", lambda sid: redirected.append(sid))

    sid = _seed_call(db, "CA-human-test")

    resp = api_client.post(
        "/twilio/amd_status",
        data={"CallSid": sid, "AnsweredBy": "human"},
    )
    assert resp.status_code == 200
    assert redirected == []

    log = db.query(CallLog).filter_by(twilio_call_id=sid).first()
    assert log.status == "queued"
