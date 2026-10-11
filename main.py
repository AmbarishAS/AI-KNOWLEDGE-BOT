"""Convenience entrypoint.

The real application lives in ``app.main:app``. Run the API with:

    uvicorn app.main:app --reload

This shim lets ``python main.py`` also start the server for quick local runs.
"""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
