def test_demo_page_served(api_client):
    resp = api_client.get("/demo")
    assert resp.status_code == 200
    assert "Voice Agent" in resp.text
    # Core pieces of the browser audio pipeline
    assert "getUserMedia" in resp.text or "mediaDevices" in resp.text
    assert "/ws" in resp.text


def test_demo_page_is_not_admin_protected(api_client):
    """Demo page must load without X-API-Key - it's read-only UI."""
    resp = api_client.get("/demo")  # no auth headers on purpose
    assert resp.status_code == 200
