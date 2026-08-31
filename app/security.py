import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any

import httpx
import jwt as pyjwt
from cryptography.fernet import Fernet
from jose import JWTError, jwt
from jwt import PyJWKSet
from passlib.context import CryptContext

from app.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
settings = get_settings()
_jwks_cache: dict[str, Any] | None = None


def _fetch_supabase_jwks() -> dict[str, Any]:
    """Load JWKS for ES256 / asymmetric Supabase signing keys (httpx for reliable TLS)."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache
    base = (settings.supabase_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("SUPABASE_URL is not configured")
    response = httpx.get(f"{base}/auth/v1/.well-known/jwks.json", timeout=15.0)
    response.raise_for_status()
    _jwks_cache = response.json()
    return _jwks_cache


def _signing_key_for_token(token: str):
    header = pyjwt.get_unverified_header(token)
    kid = header.get("kid")
    jwks = PyJWKSet.from_dict(_fetch_supabase_jwks())
    if kid:
        for key in jwks.keys:
            if key.key_id == kid:
                return key
    if jwks.keys:
        return jwks.keys[0]
    raise ValueError("No JWKS signing keys available")


def _fernet() -> Fernet:
    raw = settings.encryption_key.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
    return Fernet(key)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    return pwd_context.verify(plain, hashed)


def create_access_token(subject: str, extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"sub": subject, "exp": datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """Decode legacy app-issued JWT (SECRET_KEY). Prefer decode_supabase_token for Auth."""
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:
        raise ValueError("Invalid token") from exc


def decode_supabase_token(token: str) -> dict[str, Any]:
    """Verify a Supabase Auth access token.

    New projects often sign with ES256 (JWT signing keys / JWKS).
    Older projects use HS256 + SUPABASE_JWT_SECRET.
    """
    global _jwks_cache
    try:
        header = pyjwt.get_unverified_header(token)
    except Exception as exc:
        raise ValueError("Invalid Supabase token") from exc

    alg = (header.get("alg") or "").upper()
    base = (settings.supabase_url or "").strip().rstrip("/")
    issuer = f"{base}/auth/v1" if base else None

    # Asymmetric signing keys (default on many new Supabase projects)
    if alg in {"ES256", "RS256"} or header.get("kid"):
        try:
            signing_key = _signing_key_for_token(token)
            return pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256", "RS256"],
                audience="authenticated",
                issuer=issuer,
            )
        except Exception as exc:
            _jwks_cache = None  # retry fresh JWKS next time
            raise ValueError("Invalid Supabase token") from exc

    # Legacy shared JWT secret (HS256)
    secret = settings.resolved_jwt_secret()
    if not secret:
        raise ValueError("SUPABASE_JWT_SECRET is not configured")
    try:
        return jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except JWTError as exc:
        raise ValueError("Invalid Supabase token") from exc


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify Supabase Auth access token or local app JWT."""
    try:
        return decode_supabase_token(token)
    except Exception:
        pass
    try:
        return decode_token(token)
    except Exception as exc:
        raise ValueError("Invalid access token") from exc


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_secret(token: str) -> str:
    return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")


def mask_key(value: str) -> str:
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def generate_api_key() -> tuple[str, str, str]:
    raw = f"biqs_{secrets.token_urlsafe(32)}"
    prefix = raw[:12]
    hashed = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return raw, prefix, hashed


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:100] or secrets.token_hex(4)
