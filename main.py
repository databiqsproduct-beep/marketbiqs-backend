"""Railway / local ASGI entrypoint."""

from pathlib import Path
from dotenv import load_dotenv
import uvicorn

# Load .env from this package root (works locally and on Railway)
load_dotenv(Path(__file__).resolve().parent / ".env")

from app.main import app  # noqa: E402, F401

if __name__ == "__main__":
    uvicorn.run("app.main:main", host="127.0.0.1", port=8000, reload=True)
