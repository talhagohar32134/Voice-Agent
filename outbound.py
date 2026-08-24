import logging
import urllib.parse

from twilio.rest import Client

import config

logger = logging.getLogger(__name__)

twilio_client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)


def initiate_outbound_call(to_number: str):
    """
    Initiate an outbound call with Twilio using AMD (Answering Machine Detection).
    Async AMD means our /twilio/outbound webhook fires immediately on answer and
    AMD results arrive separately at /twilio/amd_status.
    """
    url = urllib.parse.urljoin(config.BASE_URL, "/twilio/outbound")
    amd_callback = urllib.parse.urljoin(config.BASE_URL, "/twilio/amd_status")
    status_callback = urllib.parse.urljoin(config.BASE_URL, "/twilio/status")

    call = twilio_client.calls.create(
        to=to_number,
        from_=config.TWILIO_PHONE_NUMBER,
        url=url,
        machine_detection="Enable",
        async_amd="true",
        async_amd_status_callback=amd_callback,
        status_callback=status_callback,
        status_callback_event=["completed", "busy", "failed", "no-answer"],
    )
    return call.sid


def redirect_call_to_voicemail(call_sid: str):
    """
    Once async AMD flags a machine, redirect the live call to voicemail TwiML.
    Safe to call from the AMD webhook context.
    """
    try:
        twilio_client.calls(call_sid).update(twiml=get_voicemail_twiml_text())
        logger.info("Call %s redirected to voicemail.", call_sid)
    except Exception:
        logger.exception("Failed to redirect call %s to voicemail", call_sid)


def get_voicemail_twiml_text() -> str:
    # Imported lazily to keep telephony <-> outbound decoupled
    from telephony import get_voicemail_twiml

    return get_voicemail_twiml()
