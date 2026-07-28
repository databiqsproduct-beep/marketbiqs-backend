from datetime import datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import ClientBrand, DeliveryLog, Report
from app.services import ai as ai_service

settings = get_settings()


async def deliver_update(
    db: AsyncSession,
    agency_id: str,
    agency_name: str,
    client: ClientBrand,
    report: Report | None = None,
    channel: str | None = None,
    custom_message: str | None = None,
) -> list[DeliveryLog]:
    channel = channel or client.delivery_channel.value
    if report:
        body = custom_message or (
            f"{agency_name} update for {client.name}\n\n"
            f"{report.title}\n\n{report.summary[:1200]}"
        )
        subject = report.title
        report_id = report.id
    else:
        body = custom_message or f"{agency_name}: latest intelligence pulse for {client.name}."
        subject = f"{client.name} intelligence update"
        report_id = None

    logs: list[DeliveryLog] = []
    if channel in ("email", "both"):
        for email in client.delivery_emails or []:
            status, detail = await _send_email(email, subject, body)
            log = DeliveryLog(
                agency_id=agency_id,
                client_id=client.id,
                report_id=report_id,
                channel="email",
                recipient=email,
                status=status,
                detail=detail,
            )
            db.add(log)
            logs.append(log)

    if channel in ("whatsapp", "both") and client.delivery_whatsapp:
        status, detail = await _send_whatsapp(client.delivery_whatsapp, body)
        log = DeliveryLog(
            agency_id=agency_id,
            client_id=client.id,
            report_id=report_id,
            channel="whatsapp",
            recipient=client.delivery_whatsapp,
            status=status,
            detail=detail,
        )
        db.add(log)
        logs.append(log)

    client.last_delivered_at = datetime.utcnow()
    await db.flush()
    return logs


async def _send_email(to: str, subject: str, body: str) -> tuple[str, str]:
    if not settings.resend_api_key:
        return "queued_local", "Resend not configured; delivery logged locally"
    try:
        import resend

        resend.api_key = settings.resend_api_key
        resend.Emails.send(
            {
                "from": settings.email_from,
                "to": [to],
                "subject": subject,
                "text": body,
            }
        )
        return "sent", "Delivered via Resend"
    except Exception as exc:
        return "failed", str(exc)[:400]


async def _send_whatsapp(to: str, body: str) -> tuple[str, str]:
    if not settings.twilio_account_sid or not settings.twilio_auth_token or not settings.twilio_whatsapp_from:
        return "queued_local", "Twilio WhatsApp not configured; delivery logged locally"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{settings.twilio_account_sid}/Messages.json"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                data={
                    "From": f"whatsapp:{settings.twilio_whatsapp_from}",
                    "To": f"whatsapp:{to}",
                    "Body": body[:1500],
                },
                auth=(settings.twilio_account_sid, settings.twilio_auth_token),
            )
            if response.status_code >= 400:
                return "failed", response.text[:400]
            return "sent", "Delivered via Twilio WhatsApp"
    except Exception as exc:
        return "failed", str(exc)[:400]


async def draft_delivery_copy(db: AsyncSession, agency_id: str, client_name: str, summary: str) -> str:
    return await ai_service.chat_completion(
        db,
        agency_id,
        "Write a concise client update email from a marketing agency. Professional, actionable, under 180 words.",
        f"Client: {client_name}\nSummary:\n{summary}",
    )
