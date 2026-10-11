"""Load the bundled sample documents into a running Copilot API.

Usage (API must be running):
    python -m scripts.load_sample_docs --url http://localhost:8000 --api-key changeme-dev-key

Uploads each file in data/sample_docs so you can immediately try the chat/evaluation flows.
"""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path

import httpx

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload sample documents to the Copilot API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="changeme-dev-key")
    parser.add_argument("--tenant", default="public")
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key, "X-Tenant": args.tenant}
    files = sorted(SAMPLE_DIR.glob("*"))
    if not files:
        print(f"No sample documents found in {SAMPLE_DIR}")
        return

    with httpx.Client(base_url=args.url, timeout=30.0) as client:
        for path in files:
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            with path.open("rb") as fh:
                resp = client.post(
                    "/api/v1/documents",
                    headers=headers,
                    files={"file": (path.name, fh, ctype)},
                )
            if resp.status_code == 201:
                data = resp.json()
                print(f"[ok] {path.name} -> {data['document_id']} ({data['num_chunks']} chunks)")
            else:
                print(f"[fail] {path.name}: {resp.status_code} {resp.text}")


if __name__ == "__main__":
    main()
