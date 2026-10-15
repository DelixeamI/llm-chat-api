"""Idempotency keys: a retried request must not repeat its effects.

The problem is not the client's bug but the network's. The server may finish the work and the
response may still never arrive (connection dropped, proxy timeout, client crashed after the
send). The client cannot tell that apart from "the server never got it", so it retries, and
without protection a retry of a chat request means a second generation, a second set of rows
and a second charge.

The client attaches a unique `Idempotency-Key` to the request. The first request with a key
runs; the response is stored. Any later request with the same key gets the stored response
back and nothing runs again.
"""

import hashlib
import json
import logging
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.kv.client import make_key
from app.kv.cooldown import Cooldown

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 30.0


@dataclass(frozen=True)
class Proceed:
    """First request with this key: run it, then call complete()."""


@dataclass(frozen=True)
class Replay:
    """A finished request with this key exists: return its stored response."""

    body: str


@dataclass(frozen=True)
class InProgress:
    """The first request with this key is still running."""


@dataclass(frozen=True)
class Mismatch:
    """The key was already used for a different request: a client bug, not a retry."""


@dataclass(frozen=True)
class Unprotected:
    """Redis is unavailable: run the request without idempotency."""


Begin = Proceed | Replay | InProgress | Mismatch | Unprotected


def fingerprint(payload: str) -> str:
    """What makes two requests 'the same': a hash of the canonical request body."""
    canonical = json.dumps(
        json.loads(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class IdempotencyStore:
    def __init__(self, redis: Redis, *, ttl_seconds: int, lock_seconds: int) -> None:
        self._redis = redis
        # How long a finished response is kept: the window in which a retry is recognised.
        # Retries come within seconds or minutes, so a day is generous; it also bounds how
        # long a stored response (which may repeat user data) lives in Redis.
        self._ttl_seconds = ttl_seconds
        # How long an "in progress" marker lives. It must outlast the slowest legitimate
        # request, or a retry that arrives mid-request would start a second run; and it must
        # expire on its own, or a crashed process would block its key forever.
        self._lock_seconds = lock_seconds
        self._cooldown = Cooldown(COOLDOWN_SECONDS)

    def _key(self, client: str, key: str) -> str:
        # The client is part of the key: two clients may pick the same idempotency key, and
        # one must never be handed the other's stored response. The key itself is hashed so
        # that whatever the client sent cannot shape a Redis key.
        return make_key("idem", client, hashlib.sha256(key.encode()).hexdigest())

    async def begin(self, client: str, key: str, request_fingerprint: str) -> Begin:
        if self._cooldown.active():
            return Unprotected()
        redis_key = self._key(client, key)
        marker = json.dumps({"fingerprint": request_fingerprint, "state": "in_progress"})
        try:
            # SET NX is the whole mutual exclusion: one command, so of several simultaneous
            # requests with the same key exactly one gets True and runs
            if await self._redis.set(redis_key, marker, nx=True, ex=self._lock_seconds):
                return Proceed()
            raw = await self._redis.get(redis_key)
            if raw is None:
                # The previous entry expired between the two commands: try to take the key
                if await self._redis.set(redis_key, marker, nx=True, ex=self._lock_seconds):
                    return Proceed()
                raw = await self._redis.get(redis_key)
        except RedisError as exc:
            self._fail("begin", exc)
            return Unprotected()
        if raw is None:
            return InProgress()
        stored = json.loads(raw)
        if stored["fingerprint"] != request_fingerprint:
            return Mismatch()
        if stored["state"] == "done":
            return Replay(body=stored["body"])
        return InProgress()

    async def complete(self, client: str, key: str, request_fingerprint: str, body: str) -> None:
        value = json.dumps({"fingerprint": request_fingerprint, "state": "done", "body": body})
        try:
            await self._redis.set(self._key(client, key), value, ex=self._ttl_seconds)
        except RedisError as exc:
            self._fail("complete", exc)

    async def abort(self, client: str, key: str) -> None:
        """The request failed: free the key so the client's retry can actually run."""
        try:
            await self._redis.delete(self._key(client, key))
        except RedisError as exc:
            self._fail("abort", exc)

    def _fail(self, operation: str, exc: RedisError) -> None:
        self._cooldown.trip()
        logger.warning(
            "Idempotency %s failed, requests run unprotected for %.0fs: %s",
            operation,
            self._cooldown.seconds,
            type(exc).__name__,
        )
