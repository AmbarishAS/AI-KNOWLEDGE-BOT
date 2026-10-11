"""In-process metrics registry.

A tiny, dependency-free counter/timer registry so the app can expose basic throughput,
latency, token, and cost metrics without pulling in Prometheus for a local demo. The design
is OpenTelemetry-friendly: swap these for OTel instruments later without touching callers.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._timers: dict[str, list[float]] = defaultdict(list)

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, millis: float) -> None:
        with self._lock:
            self._timers[name].append(millis)

    def snapshot(self) -> dict:
        with self._lock:
            timers = {}
            for k, vals in self._timers.items():
                if vals:
                    ordered = sorted(vals)
                    timers[k] = {
                        "count": len(vals),
                        "avg_ms": round(sum(vals) / len(vals), 2),
                        "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1], 2),
                        "max_ms": round(max(vals), 2),
                    }
            return {"counters": dict(self._counters), "timers": timers}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timers.clear()


METRICS = Metrics()
