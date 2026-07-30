from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Integration, JiraTicket
from app.security import decrypt_secret, encrypt_secret


async def connect_jira(
    db: AsyncSession,
    agency_id: str,
    base_url: str,
    email: str,
    api_token: str,
    project_key: str,
    epic_name_field: str | None = None,
) -> Integration:
    base = str(base_url).rstrip("/")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{base}/rest/api/3/myself",
            auth=(email, api_token),
        )
        if response.status_code >= 400:
            raise ValueError("Unable to authenticate with Jira. Check URL, email, and API token.")

    stmt = select(Integration).where(Integration.agency_id == agency_id, Integration.provider == "jira")
    result = await db.execute(stmt)
    integration = result.scalar_one_or_none()
    credentials = encrypt_secret(f"{email}::{api_token}")
    config = {
        "base_url": base,
        "email": email,
        "project_key": project_key,
        "epic_name_field": epic_name_field or "customfield_10011",
    }
    if not integration:
        integration = Integration(
            agency_id=agency_id,
            provider="jira",
            config=config,
            encrypted_credentials=credentials,
            is_connected=True,
        )
        db.add(integration)
    else:
        integration.config = config
        integration.encrypted_credentials = credentials
        integration.is_connected = True
    await db.flush()
    return integration


async def create_jira_ticket(
    db: AsyncSession,
    agency_id: str,
    client_id: str,
    title: str,
    description: str,
    insight_id: str | None = None,
    issue_type: str = "Task",
    parent_epic_key: str | None = None,
) -> JiraTicket:
    stmt = select(Integration).where(
        Integration.agency_id == agency_id,
        Integration.provider == "jira",
        Integration.is_connected.is_(True),
    )
    result = await db.execute(stmt)
    integration = result.scalar_one_or_none()
    if not integration or not integration.encrypted_credentials:
        raise ValueError("Connect your Jira account first.")

    email, token = decrypt_secret(integration.encrypted_credentials).split("::", 1)
    base = integration.config.get("base_url")
    project_key = integration.config.get("project_key")
    epic_name_field = integration.config.get("epic_name_field") or "customfield_10011"

    fields: dict[str, Any] = {
        "project": {"key": project_key},
        "summary": title[:240],
        "description": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description[:5000]}],
                }
            ],
        },
        "issuetype": {"name": issue_type},
    }
    if issue_type.lower() == "epic":
        fields[epic_name_field] = title[:240]
    if parent_epic_key and issue_type.lower() != "epic":
        fields["parent"] = {"key": parent_epic_key}

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            f"{base}/rest/api/3/issue",
            auth=(email, token),
            json={"fields": fields},
        )
        if response.status_code >= 400 and parent_epic_key:
            fields.pop("parent", None)
            fields["issuetype"] = {"name": "Task"}
            response = await client.post(
                f"{base}/rest/api/3/issue",
                auth=(email, token),
                json={"fields": fields},
            )
        if response.status_code >= 400 and issue_type.lower() != "task":
            fields["issuetype"] = {"name": "Task"}
            fields.pop(epic_name_field, None)
            response = await client.post(
                f"{base}/rest/api/3/issue",
                auth=(email, token),
                json={"fields": fields},
            )
        if response.status_code >= 400:
            raise ValueError(f"Jira create failed: {response.text[:400]}")
        data = response.json()

    ticket = JiraTicket(
        agency_id=agency_id,
        client_id=client_id,
        insight_id=insight_id,
        jira_key=data.get("key"),
        jira_url=f"{base}/browse/{data.get('key')}" if data.get("key") else None,
        title=title,
        description=description,
        status="created",
    )
    db.add(ticket)
    await db.flush()
    return ticket
