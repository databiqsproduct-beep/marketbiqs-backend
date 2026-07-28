"""Railway / local ASGI entrypoint."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from this package root (works locally and on Railway)
load_dotenv(Path(__file__).resolve().parent / ".env")

from app.main import app  # noqa: E402, F401
