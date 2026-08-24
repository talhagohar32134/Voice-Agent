from telephony import get_inbound_twiml, get_outbound_twiml, get_voicemail_twiml


def test_inbound_twiml_converts_to_wss():
    twiml = get_inbound_twiml("https://example.ngrok.app")
    assert "<Connect>" in twiml
    assert "wss://example.ngrok.app/ws" in twiml
    assert "inbound_and_outbound" in twiml


def test_inbound_twiml_plain_http_becomes_ws():
    twiml = get_inbound_twiml("http://localhost:8000")
    assert "ws://localhost:8000/ws" in twiml


def test_outbound_twiml_marks_is_outbound():
    twiml = get_outbound_twiml("https://example.ngrok.app")
    assert 'name="is_outbound"' in twiml
    assert 'value="true"' in twiml
    assert "wss://example.ngrok.app/ws" in twiml


def test_voicemail_twiml_pauses_speaks_hangup():
    twiml = get_voicemail_twiml()
    assert "<Pause" in twiml  # wait out the voicemail beep
    assert "<Say" in twiml
    assert "<Hangup" in twiml
