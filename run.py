from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from app.main import app  # noqa: E402, F401
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

# Load .env from this package root
load_dotenv(Path(__file__).resolve().parent / ".env")

from app.main import app  # noqa: E402, F401

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)