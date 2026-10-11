"""Token counting and cost estimation.

Token counting matters for (a) staying within the context window and (b) estimating cost.
We use ``tiktoken`` when available for accurate counts, and fall back to a ~4 chars/token
heuristic otherwise. Prices are a small, editable table — document that they change and must
be verified against the provider's current pricing.
"""

from __future__ import annotations

from functools import lru_cache

# USD per 1K tokens (illustrative; verify against provider pricing). input, output.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
    "mock": (0.0, 0.0),
}


@lru_cache
def _encoder(model: str):
    try:
        import tiktoken

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = _encoder(model)
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    inp, out = _PRICES.get(model, (0.0, 0.0))
    return round((prompt_tokens / 1000) * inp + (completion_tokens / 1000) * out, 6)
