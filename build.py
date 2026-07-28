"""Production build check for MarketBiqs backend."""

from __future__ import annotations

import compileall
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ok = compileall.compile_dir(str(ROOT / "app"), quiet=1)
    ok = compileall.compile_dir(str(ROOT / "scripts"), quiet=1) and ok
    if not ok:
        print("compileall failed", file=sys.stderr)
        return 1
    # Import app (validates wiring)
    sys.path.insert(0, str(ROOT))
    from app.main import app  # noqa: F401

    print("backend build ok:", app.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
