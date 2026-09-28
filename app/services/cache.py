"""Cache of model replies for requests whose answer does not depend on chance."""

import hashlib
import json
import logging
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.kv.client import make_key
from app.llm.base import Completion
from app.schemas.cache import CacheStats
from app.schemas.chat import GenerationParams, Message

logger = logging.getLogger(__name__)

# Part of every key. Bump it when the stored format or the key recipe changes: old entries
# become unreachable and expire on their own.
CACHE_VERSION = "v1"
# After a Redis error the cache is skipped for this long. Without it, every request during
# an outage would wait for two timeouts (lookup and store) and log two warnings.
COOLDOWN_SECONDS = 30.0

HITS_KEY = make_key("stats", "cache", "hits")
MISSES_KEY = make_key("stats", "cache", "misses")


def is_cacheable(params: GenerationParams) -> bool:
    # Only greedy decoding (temperature 0) picks the most likely token every time. With
    # sampling each call may return a different answer, and replaying one stored sample
    # would silently take that variety away from the client who asked for it.
    return params.temperature == 0


class ResponseCache:
    def __init__(self, redis: Redis, *, ttl_seconds: int, scope: str) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds
        # Settings outside the request that also shape the answer, such as the provider
        # and its reasoning mode; changing them must not serve replies made under old ones
        self._scope = scope
        self._skip_until = 0.0

    def key_for(self, model: str, messages: list[Message], params: GenerationParams) -> str:
        """Key from everything that determines the reply; any difference gives a new key."""
        # ponytail: shared by all clients. Once API keys exist, decide whether to scope it per
        # client: a fast reply reveals that someone already sent the same conversation.
        payload = {
            "scope": self._scope,
            "model": model,
            "messages": [message.model_dump() for message in messages],
            # The whole model rather than chosen fields: a parameter added later changes
            # the key without anyone having to remember to include it here
            "params": params.model_dump(),
        }
        # sort_keys and fixed separators: equal payloads always serialize to equal bytes
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return make_key("cache", CACHE_VERSION, digest)

    async def get(self, key: str) -> Completion | None:
        """Stored reply, or None on a miss. Never raises: without Redis the model answers."""
        if not self._available():
            return None
        try:
            raw = await self._redis.get(key)
            await self._redis.incr(MISSES_KEY if raw is None else HITS_KEY)
        except RedisError as exc:
            self._fail("lookup", exc)
            return None
        if raw is None:
            return None
        try:
            return Completion(**json.loads(raw), cached=True)
        except (ValueError, TypeError) as exc:
            # A damaged or foreign value is a miss, not an error for the client
            logger.warning("Ignoring unreadable cache entry: %s", type(exc).__name__)
            return None

    async def put(self, key: str, completion: Completion) -> None:
        if not self._available():
            return
        value = json.dumps(
            {
                "content": completion.content,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "finish_reason": completion.finish_reason,
            },
            ensure_ascii=False,
        )
        try:
            # Value and TTL in one command: SET followed by EXPIRE would leave a key that
            # never expires if the process died between the two
            await self._redis.set(key, value, ex=self._ttl_seconds)
        except RedisError as exc:
            self._fail("store", exc)

    async def stats(self) -> CacheStats:
        hits, misses = await self._redis.mget(HITS_KEY, MISSES_KEY)
        return CacheStats(enabled=True, hits=int(hits or 0), misses=int(misses or 0))

    def _available(self) -> bool:
        return time.monotonic() >= self._skip_until

    def _fail(self, operation: str, exc: RedisError) -> None:
        self._skip_until = time.monotonic() + COOLDOWN_SECONDS
        logger.warning(
            "Cache %s failed, bypassing cache for %.0fs: %s",
            operation,
            COOLDOWN_SECONDS,
            type(exc).__name__,
        )
