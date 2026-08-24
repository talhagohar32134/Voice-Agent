from twilio.twiml.voice_response import Connect, Stream, VoiceResponse

import config


def _stream_ws_url(base_url: str) -> str:
    return base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"


def get_inbound_twiml(host: str) -> str:
    """Generate TwiML for an incoming call to connect to the WebSocket."""
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=_stream_ws_url(host), track="inbound_and_outbound")
    response.append(connect)
    return str(response)


def get_outbound_twiml(host: str) -> str:
    """Generate TwiML for an outbound call (when a human answers) to connect to the WebSocket."""
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=_stream_ws_url(host), track="inbound_and_outbound")
    # Custom parameter so the WS knows this is an outbound call that just connected
    stream.parameter(name="is_outbound", value="true")
    connect.append(stream)
    response.append(connect)
    return str(response)


def get_voicemail_twiml() -> str:
    """Generate TwiML to leave a voicemail message."""
    response = VoiceResponse()
    pause_seconds = 2
    response.pause(length=pause_seconds)  # wait out the voicemail beep
    response.say(config.VOICEMAIL_MESSAGE)
    response.hangup()
    return str(response)
