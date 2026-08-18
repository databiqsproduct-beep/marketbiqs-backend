import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from groq import AsyncGroq, Groq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import ApiKeyVault
from app.security import decrypt_secret

settings = get_settings()
logger = logging.getLogger("marketbiqs.ai")

# Groq retired llama-3.3-70b-versatile on 2026-08-16.
_DEFAULT_GROQ_MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b")
FALLBACK_PREFIX = "Competitive intelligence briefing prepared from available workspace data."


def _chat_models() -> list[str]:
    configured = (getattr(settings, "groq_model", None) or "").strip()
    models: list[str] = []
    if configured:
        models.append(configured)
    for model in _DEFAULT_GROQ_MODELS:
        if model not in models:
            models.append(model)
    return models


def _is_missing_model(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "model_not_found" in text or "does not exist or you do not have access" in text


CHAT_MODEL = _chat_models()[0]


async def resolve_groq_key(db: AsyncSession, agency_id: str) -> str:
    stmt = select(ApiKeyVault).where(
        ApiKeyVault.agency_id == agency_id,
        ApiKeyVault.provider == "groq",
        ApiKeyVault.is_active.is_(True),
    )
    result = await db.execute(stmt)
    vault = result.scalar_one_or_none()
    if vault:
        return decrypt_secret(vault.encrypted_key)
    return settings.groq_api_key


def _client(api_key: str) -> Groq | None:
    if not api_key:
        return None
    return Groq(api_key=api_key)


def _async_client(api_key: str) -> AsyncGroq | None:
    if not api_key:
        return None
    return AsyncGroq(api_key=api_key)


def _normalize_messages(
    system: str,
    user: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for item in history or []:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


def is_fallback_text(text: str | None) -> bool:
    return bool(text) and text.strip().startswith(FALLBACK_PREFIX)


async def chat_completion(
    db: AsyncSession,
    agency_id: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    history: list[dict[str, str]] | None = None,
    *,
    json_mode: bool = False,
) -> str:
    api_key = await resolve_groq_key(db, agency_id)
    client = _async_client(api_key)
    if not client:
        logger.warning("Groq key missing for agency=%s — using fallback text", agency_id)
        return _fallback_text(system, user)
    messages = _normalize_messages(system, user, history)
    last_exc: Exception | None = None
    for model in _chat_models():
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = await client.chat.completions.create(**kwargs)
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            logger.warning("Groq returned empty content for agency=%s model=%s", agency_id, model)
        except Exception as exc:
            last_exc = exc
            if _is_missing_model(exc):
                logger.warning("Groq model unavailable (%s); trying next", model)
                continue
            if json_mode:
                logger.warning("Groq json_mode failed for agency=%s model=%s (%s); retrying plain", agency_id, model, exc)
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    )
                    content = (response.choices[0].message.content or "").strip()
                    if content:
                        return content
                except Exception as exc2:
                    last_exc = exc2
                    if _is_missing_model(exc2):
                        logger.warning("Groq model unavailable (%s); trying next", model)
                        continue
                    logger.exception("Groq chat_completion failed for agency=%s: %s", agency_id, exc2)
                    return _fallback_text(system, user)
            else:
                logger.exception("Groq chat_completion failed for agency=%s: %s", agency_id, exc)
                return _fallback_text(system, user)
    if last_exc:
        logger.exception("Groq chat_completion failed for agency=%s: %s", agency_id, last_exc)
    return _fallback_text(system, user)


async def chat_completion_stream(
    db: AsyncSession,
    agency_id: str,
    system: str,
    user: str,
    temperature: float = 0.45,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Yield text deltas from Groq. Falls back to a single chunk if streaming fails."""
    api_key = await resolve_groq_key(db, agency_id)
    client = _async_client(api_key)
    if not client:
        yield _fallback_text(system, user)
        return
    try:
        stream = await client.chat.completions.create(
            model=_chat_models()[0],
            messages=_normalize_messages(system, user, history),
            temperature=temperature,
            stream=True,
        )
        produced = False
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                produced = True
                yield delta
        if not produced:
            yield _fallback_text(system, user)
    except Exception as exc:
        logger.exception("Groq stream failed for agency=%s: %s", agency_id, exc)
        text = await chat_completion(db, agency_id, system, user, temperature=temperature, history=history)
        yield text


async def platform_chat_completion_stream(
    system: str,
    user: str,
    temperature: float = 0.35,
    history: list[dict[str, str]] | None = None,
    *,
    missing_key_message: str | None = None,
) -> AsyncIterator[str]:
    """Stream from the platform GROQ_API_KEY only (help desk / product support — ignores BYOK)."""
    api_key = (settings.groq_api_key or "").strip()
    client = _async_client(api_key)
    if not client:
        yield missing_key_message or _fallback_text(system, user)
        return
    try:
        stream = await client.chat.completions.create(
            model=_chat_models()[0],
            messages=_normalize_messages(system, user, history),
            temperature=temperature,
            stream=True,
        )
        produced = False
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                produced = True
                yield delta
        if not produced:
            yield "I could not generate a reply right now. Please try again in a moment."
    except Exception as exc:
        logger.exception("Platform Groq stream failed: %s", exc)
        yield (
            "Help desk hit a Groq error. Check that GROQ_API_KEY is valid, then try again. "
            f"({type(exc).__name__})"
        )


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text or is_fallback_text(text):
        return None
    # Prefer fenced JSON if present
    if "```" in text:
        start_fence = text.find("```")
        chunk = text[start_fence + 3 :]
        if chunk.lstrip().startswith("json"):
            chunk = chunk.lstrip()[4:]
        end_fence = chunk.find("```")
        if end_fence >= 0:
            text = chunk[:end_fence].strip()
    try:
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj >= 0 and end_obj > start_obj:
            parsed = json.loads(text[start_obj : end_obj + 1])
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        return None
    return None


async def structured_json(
    db: AsyncSession,
    agency_id: str,
    system: str,
    user: str,
    temperature: float = 0.25,
) -> dict[str, Any]:
    raw = await chat_completion(
        db,
        agency_id,
        (
            system
            + " Respond with a single valid JSON object only. "
            "All string values must use double quotes. No markdown fences. No trailing commentary."
        ),
        user,
        temperature=temperature,
        json_mode=True,
    )
    parsed = _extract_json_object(raw)
    if parsed is not None:
        return parsed

    # One repair attempt when the model drifts from valid JSON
    if raw and not is_fallback_text(raw):
        repair = await chat_completion(
            db,
            agency_id,
            (
                "Fix the following into a single valid JSON object. "
                "Keep the same meaning. Output JSON only with double-quoted strings."
            ),
            raw[:8000],
            temperature=0,
            json_mode=True,
        )
        parsed = _extract_json_object(repair)
        if parsed is not None:
            return parsed

    logger.warning("structured_json parse failed for agency=%s raw_head=%r", agency_id, (raw or "")[:180])
    return {}


def _fallback_text(system: str, user: str) -> str:
    return (
        f"{FALLBACK_PREFIX} "
        f"Focus: {user[:400]}. Guidance: {system[:200]}"
    )
