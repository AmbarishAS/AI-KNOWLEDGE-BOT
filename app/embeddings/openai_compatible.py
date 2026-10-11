"""OpenAI-compatible embedding provider (real mode).

Uses httpx against any OpenAI-compatible ``/embeddings`` endpoint. Implements:
- batching (sending many texts per request),
- bounded retries with exponential backoff for transient errors/rate limits,
- a request timeout.

This is only used when ``APP_MODE=real``; tests never hit the network.
"""

from __future__ import annotations

import asyncio

import httpx

from app.embeddings.base import EmbeddingProvider
from app.logging_config import get_logger

logger = get_logger(__name__)


class OpenAICompatibleEmbeddings(EmbeddingProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 64,
    ) -> None:
        self.dim = dim
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._max_retries = max_retries
        self._batch_size = batch_size

    async def _post(self, inputs: list[str]) -> list[list[float]]:
        payload = {"model": self._model, "input": inputs}
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(f"{self._base_url}/embeddings", json=payload, headers=self._headers)
                    if resp.status_code in (429, 500, 502, 503):
                        raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                    resp.raise_for_status()
                    data = resp.json()["data"]
                    return [item["embedding"] for item in data]
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_exc = exc
                backoff = 2**attempt
                logger.warning(f"embedding retry {attempt + 1}/{self._max_retries}: {exc}")
                await asyncio.sleep(backoff * 0.1)
        raise RuntimeError(f"Embedding request failed after retries: {last_exc}")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(await self._post(batch))
        return results

    async def embed_query(self, text: str) -> list[float]:
        return (await self._post([text]))[0]
