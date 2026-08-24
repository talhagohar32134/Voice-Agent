import asyncio
import logging
from datetime import datetime

import phonenumbers
import pytz
from phonenumbers import timezone

from database import CallLog, CallQueue, ConsentRecord, DoNotCall, SessionLocal
from outbound import initiate_outbound_call

import config

logger = logging.getLogger(__name__)


def is_within_calling_hours(phone_num: str) -> bool:
    """TCPA-style guard: only call during local daytime hours at the callee."""
    try:
        parsed = phonenumbers.parse(phone_num, None)
        zones = timezone.time_zones_for_number(parsed)

        if not zones or zones[0] == "Etc/Unknown":
            # Unknown timezone - be safe and deny.
            return False

        tz = pytz.timezone(zones[0])
        local_hour = datetime.now(tz).hour
        return config.CALLING_HOUR_START <= local_hour < config.CALLING_HOUR_END
    except Exception:
        logger.exception("Timezone check failed for %s", phone_num)
        return False


def can_call_number(db, phone_num: str) -> bool:
    if db.query(DoNotCall).filter(DoNotCall.phone_number == phone_num).first():
        return False

    consent = db.query(ConsentRecord).filter(ConsentRecord.phone_number == phone_num).first()
    if not consent or not consent.opt_in_timestamp:
        return False

    if not is_within_calling_hours(phone_num):
        return False

    return True


async def process_queue_batch():
    """Process pending queue entries with policy checks (DNC / consent / hours)."""
    def _fetch_pending():
        db = SessionLocal()
        try:
            return [
                (item.id, item.phone_number)
                for item in db.query(CallQueue).filter(CallQueue.status == "pending").all()
            ]
        finally:
            db.close()

    pending = await asyncio.to_thread(_fetch_pending)
    logger.info("Processing %d queued calls", len(pending))

    for item_id, phone_num in pending:

        def _process_one():
            db = SessionLocal()
            try:
                item = db.get(CallQueue, item_id)
                if not item or item.status != "pending":
                    return
                if not can_call_number(db, phone_num):
                    item.status = "skipped_policy"
                    logger.info("Skipped %s (policy)", phone_num)
                else:
                    try:
                        call_sid = initiate_outbound_call(phone_num)
                        db.add(CallLog(
                            twilio_call_id=call_sid,
                            direction="outbound",
                            status="queued",
                            phone_number=phone_num,
                        ))
                        item.status = "completed"
                        logger.info("Dialed %s -> %s", phone_num, call_sid)
                    except Exception as e:
                        logger.error("Failed to dial %s: %s", phone_num, e)
                        item.status = "failed"
                db.commit()
            finally:
                db.close()

        await asyncio.to_thread(_process_one)
        await asyncio.sleep(1)  # gentle pacing between dials
