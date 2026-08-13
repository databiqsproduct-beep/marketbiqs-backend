from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Integration, JiraTicket
from app.security import decrypt_secret, encrypt_secret

DEFAULT_EPIC_NAME_FIELD = "customfield_10011"


async def _fetch_project(client: httpx.AsyncClient, base: str, auth: tuple[str, str], key: str) -> dict[str, Any] | None:
    response = await client.get(f"{base}/rest/api/3/project/{key}", auth=auth)
    if response.status_code >= 400:
        return None
    return response.json()


def _mentions_epic_name_field(body: str, epic_name_field: str) -> bool:
    lowered = (body or "").lower()
    return epic_name_field.lower() in lowered or "epic name" in lowered


async def _visible_project_keys(client: httpx.AsyncClient, base: str, auth: tuple[str, str]) -> list[str]:
    response = await client.get(f"{base}/rest/api/3/project/search?maxResults=50", auth=auth)
    if response.status_code >= 400:
        return []
    return [p.get("key") for p in response.json().get("values", []) if p.get("key")]


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
    key = (project_key or "").strip().upper()
    if not key:
        raise ValueError("Enter the Jira project key, for example AIRS.")
    async with httpx.AsyncClient(timeout=30) as client:
        auth = (email, api_token)
        response = await client.get(f"{base}/rest/api/3/myself", auth=auth)
        if response.status_code >= 400:
            raise ValueError("Unable to authenticate with Jira. Check URL, email, and API token.")
        project = await _fetch_project(client, base, auth, key)
        if not project:
            visible = await _visible_project_keys(client, base, auth)
            hint = f" Projects available to you: {', '.join(visible)}." if visible else ""
            raise ValueError(
                f"Jira project '{key}' was not found, or this account cannot create issues in it.{hint}"
            )

    stmt = select(Integration).where(Integration.agency_id == agency_id, Integration.provider == "jira")
    result = await db.execute(stmt)
    integration = result.scalar_one_or_none()
    credentials = encrypt_secret(f"{email}::{api_token}")
    config = {
        "base_url": base,
        "email": email,
        "project_key": key,
        "project_style": project.get("style"),
        "epic_name_field": epic_name_field or DEFAULT_EPIC_NAME_FIELD,
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


async def disconnect_jira(db: AsyncSession, agency_id: str) -> bool:
    stmt = select(Integration).where(Integration.agency_id == agency_id, Integration.provider == "jira")
    result = await db.execute(stmt)
    integration = result.scalar_one_or_none()
    if not integration:
        return False
    integration.is_connected = False
    integration.encrypted_credentials = None
    integration.config = {
        **(integration.config or {}),
        "base_url": None,
        "email": None,
        "project_key": None,
    }
    await db.flush()
    return True


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
    project_style = integration.config.get("project_style")
    epic_name_field = integration.config.get("epic_name_field") or DEFAULT_EPIC_NAME_FIELD

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
    is_epic = issue_type.lower() == "epic"
    if parent_epic_key and not is_epic:
        fields["parent"] = {"key": parent_epic_key}

    async with httpx.AsyncClient(timeout=20) as client:
        auth = (email, token)

        if is_epic and project_style is None:
            project = await _fetch_project(client, base, auth, project_key)
            project_style = (project or {}).get("style")
            if project_style:
                integration.config = {**integration.config, "project_style": project_style}
                await db.flush()

        # Company-managed projects require the epic-name field; team-managed ones reject it.
        send_epic_name = is_epic and project_style != "next-gen"
        if send_epic_name:
            fields[epic_name_field] = title[:240]

        async def post_issue() -> httpx.Response:
            return await client.post(f"{base}/rest/api/3/issue", auth=auth, json={"fields": fields})

        response = await post_issue()

        if response.status_code >= 400 and is_epic and _mentions_epic_name_field(response.text, epic_name_field):
            if send_epic_name:
                fields.pop(epic_name_field, None)
            else:
                fields[epic_name_field] = title[:240]
            response = await post_issue()

        if response.status_code >= 400 and parent_epic_key:
            fields.pop("parent", None)
            response = await post_issue()

        if response.status_code >= 400 and issue_type.lower() != "task":
            fields["issuetype"] = {"name": "Task"}
            fields.pop(epic_name_field, None)
            response = await post_issue()

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
