"""Redis backbone for the specialist pipeline.

Redis is the coordination + memory layer, not a side cache. One :class:`JobStore`
per run owns four responsibilities:

1. **Job state** — ``job:{id}:status`` walks ``queued -> encoding -> solving ->
   executing -> aggregating -> done`` so a UI can poll progress.
2. **Shared context** — ``job:{id}:context`` holds the full structured result as the
   source of truth the API/UI reads back.
3. **Streaming** — every stage ``publish()``es an event to the ``job:{id}:events``
   pub/sub channel *and* appends to ``job:{id}:log`` so late subscribers can replay.
   Each specialist column emits an event the moment its solve finishes, so the
   parallel fan-out streams stage-by-stage to the client.
4. **Memory** — two persistent maps that make repeat runs faster and stronger:
   * ``cache:{hash}`` — code-hash+concern -> solver outcome (skip LLM + Z3 on
     unchanged code).
   * ``mem:{fn_hash}:{concern}`` — verified abstraction reuse, with a hit counter.
     The more the verifier runs on a repo, the more it reuses: improvement as a
     persisted feature.

If no Redis server is reachable it degrades to an in-process shim implementing the
same surface, so the pipeline (and the demo) never hard-fails.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from typing import Any

try:
    import redis as _redis

    _HAS_REDIS_LIB = True
except Exception:  # pragma: no cover
    _redis = None
    _HAS_REDIS_LIB = False

DEFAULT_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_JOB_TTL_S = 24 * 3600
_LOG_MAX = 500


def _now_marker(seq: int) -> str:
    # Date.now() is unavailable in some sandboxes and ordering is all we need,
    # so events are stamped with a monotonic per-run sequence, not wall-clock.
    return f"#{seq}"


class _MemoryBackend:
    """Dict-based stand-in used when no Redis server is reachable."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    def set(self, k: str, v: str, ex: int | None = None) -> None:
        self.kv[k] = v

    def get(self, k: str) -> str | None:
        return self.kv.get(k)

    def rpush(self, k: str, v: str) -> None:
        self.lists.setdefault(k, []).append(v)

    def lrange(self, k: str, a: int, b: int) -> list[str]:
        items = self.lists.get(k, [])
        return items[a:] if b == -1 else items[a : b + 1]

    def publish(self, channel: str, msg: str) -> None:
        pass  # no live subscribers in-process; the log list is the replay path

    def hincrby(self, k: str, field: str, amount: int) -> int:
        h = self.hashes.setdefault(k, {})
        h[field] = str(int(h.get(field, "0")) + amount)
        return int(h[field])

    def expire(self, k: str, ttl: int) -> None:
        pass


# Process-global in-memory store shared by all memory-mode JobStores, so a
# background worker and an SSE request in the same process see the same events
# (keys are namespaced by job_id). This makes streaming work without a Redis server.
_SHARED_MEMORY = _MemoryBackend()


class JobStore:
    """Per-run handle over Redis (or the shared in-memory shim)."""

    def __init__(self, job_id: str, url: str | None = None) -> None:
        self.job_id = job_id
        self._seq = 0
        self.backend = "memory"
        self._client: Any = _SHARED_MEMORY

        if _HAS_REDIS_LIB:
            try:
                client = _redis.Redis.from_url(url or DEFAULT_URL, decode_responses=True)
                client.ping()
                self._client = client
                self.backend = "redis"
            except Exception:
                # keep the in-memory shim
                self.backend = "memory"

    # --- keys ---
    def _k(self, suffix: str) -> str:
        return f"job:{self.job_id}:{suffix}"

    # --- job state + shared context ---
    def set_status(self, status: str) -> None:
        try:
            self._client.set(self._k("status"), status, ex=_JOB_TTL_S)
            self.publish({"stage": "status", "status": status})
        except Exception:
            pass

    def get_status(self) -> str | None:
        try:
            return self._client.get(self._k("status"))
        except Exception:
            return None

    def set_context(self, context: dict) -> None:
        try:
            self._client.set(self._k("context"), json.dumps(context, default=str), ex=_JOB_TTL_S)
        except Exception:
            pass

    def get_context(self) -> dict | None:
        try:
            raw = self._client.get(self._k("context"))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    # --- streaming (pub/sub + replayable log) ---
    def publish(self, event: dict) -> None:
        self._seq += 1
        payload = {"seq": self._seq, "marker": _now_marker(self._seq), **event}
        msg = json.dumps(payload, default=str)
        try:
            self._client.publish(self._k("events"), msg)
            self._client.rpush(self._k("log"), msg)
            self._client.expire(self._k("log"), _JOB_TTL_S)
        except Exception:
            pass

    def events(self) -> list[dict]:
        try:
            return [json.loads(m) for m in self._client.lrange(self._k("log"), 0, -1)]
        except Exception:
            return []

    def subscribe(self) -> Iterator[dict]:  # pragma: no cover - used by the live UI/API
        """Yield events in real time (Redis only; no-op generator on the shim)."""
        if self.backend != "redis":
            return
        pubsub = self._client.pubsub()
        pubsub.subscribe(self._k("events"))
        for message in pubsub.listen():
            if message.get("type") == "message":
                yield json.loads(message["data"])

    # --- caching (code-hash + concern -> outcome) ---
    @staticmethod
    def hash_code(text: str, concern: str = "") -> str:
        return hashlib.sha256(f"{concern}\x00{text}".encode()).hexdigest()[:24]

    def cache_get(self, code_hash: str) -> dict | None:
        try:
            raw = self._client.get(f"cache:{code_hash}")
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def cache_put(self, code_hash: str, outcome: dict) -> None:
        try:
            self._client.set(f"cache:{code_hash}", json.dumps(outcome, default=str))
        except Exception:
            pass

    # --- cross-run abstraction memory (the self-improvement substrate) ---
    def memory_get(self, fn_hash: str, concern: str) -> dict | None:
        try:
            raw = self._client.get(f"mem:{fn_hash}:{concern}")
            if not raw:
                return None
            self._client.hincrby("mem:hits", f"{fn_hash}:{concern}", 1)
            return json.loads(raw)
        except Exception:
            return None

    def memory_put(self, fn_hash: str, concern: str, abstraction: dict) -> None:
        try:
            self._client.set(f"mem:{fn_hash}:{concern}", json.dumps(abstraction, default=str))
        except Exception:
            pass

    def memory_hits(self, fn_hash: str, concern: str) -> int:
        try:
            return self._client.hincrby("mem:hits", f"{fn_hash}:{concern}", 0)
        except Exception:
            return 0
