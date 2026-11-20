"""``python -m openrag`` entry point."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "openrag.main:app",
        host=os.getenv("OPENRAG_HOST", "127.0.0.1"),
        port=int(os.getenv("OPENRAG_PORT", "8000")),
        reload=os.getenv("OPENRAG_RELOAD", "0").lower() in {"1", "true", "yes"},
        log_level=os.getenv("OPENRAG_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
