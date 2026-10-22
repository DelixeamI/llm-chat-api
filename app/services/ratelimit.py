"""Per-client request rate limit in Redis: a sliding window.

A rate limit counts requests per period ("20 per minute"). It is not the semaphore around the
model, which counts calls in flight at this moment. Each protects something else:

- the semaphore protects the model from too many simultaneous calls, but says nothing about
  how many requests one client sends per hour;
- the rate limit protects the budget and the other clients, but 20 requests arriving in the
  same second would still all try to run at once.

So both are needed.
"""

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.kv.client import make_key
from app.kv.cooldown import Cooldown

logger = logging.getLogger(__name__)

# One Lua script = one atomic step: Redis runs a script without interleaving other commands.
# Doing the same from Python (read the count, compare, write) would let concurrent requests
# all read "19" and all be admitted: a race that only shows up under load.
#
# Sliding window log: every admitted request is a member of a sorted set scored with its
# time. Members older than the window are dropped, the rest are counted. Unlike a fixed
# window (INCR + EXPIRE), it has no boundary burst where a client spends the limit at the
# end of one window and again at the start of the next, twice the limit within seconds.
# It costs memory proportional to the limit, which is small here (20 entries per client).
#
# The time comes from Redis itself, not from the application servers, so processes with
# slightly different clocks agree on the window.
#
# Returns {admitted (1/0), remaining, ms until the oldest request leaves the window}.
SLIDING_WINDOW = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local nonce = ARGV[3]
local t = redis.call('TIME')
local now = t[1] * 1000 + math.floor(t[2] / 1000)
redis.call('ZREMRANGEBYSCORE', key, 0, now - window_ms)
local count = redis.call('ZCARD', key)
local admitted = 0
if count < limit then
    -- the member must be unique, or two requests in the same millisecond would count once
    redis.call('ZADD', key, now, now .. ':' .. nonce)
    -- the key disappears by itself once a client goes quiet
    redis.call('PEXPIRE', key, window_ms)
    count = count + 1
    admitted = 1
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local reset_ms = tonumber(oldest[2]) + window_ms - now
return {admitted, limit - count, reset_ms}
"""

COOLDOWN_SECONDS = 30.0


@dataclass(frozen=True)
class RateLimitResult:
    admitted: bool
    limit: int
    remaining: int
    # Seconds until the oldest counted request leaves the window: when to try again if
    # rejected, and when capacity next frees up if admitted
    reset_seconds: int
    window_seconds: int

    def headers(self) -> dict[str, str]:
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset_seconds),
        }
        if not self.admitted:
            headers["Retry-After"] = str(self.reset_seconds)
        return headers


class RateLimiter:
    def __init__(self, redis: Redis, *, limit: int, window_seconds: int) -> None:
        self._script = redis.register_script(SLIDING_WINDOW)
        self._limit = limit
        self._window_seconds = window_seconds
        self._cooldown = Cooldown(COOLDOWN_SECONDS)

    async def check(self, client: str) -> RateLimitResult | None:
        """Count one request for the client. None when Redis is unavailable (see below)."""
        if self._cooldown.active():
            return None
        try:
            raw = await cast(
                Any,
                self._script(
                    keys=[make_key("ratelimit", client)],
                    args=[self._limit, self._window_seconds * 1000, uuid.uuid4().hex],
                ),
            )
        except RedisError as exc:
            # Fail open. The limiter protects a budget, not correctness, and refusing every
            # request because Redis is down would turn a cache-tier outage into a total one.
            # The semaphore around the model still bounds the damage meanwhile.
            self._cooldown.trip()
            logger.warning(
                "Rate limiter failed, not limiting for %.0fs: %s",
                self._cooldown.seconds,
                type(exc).__name__,
            )
            return None
        admitted, remaining, reset_ms = (int(value) for value in raw)
        return RateLimitResult(
            admitted=bool(admitted),
            limit=self._limit,
            remaining=max(remaining, 0),
            reset_seconds=max(math.ceil(reset_ms / 1000), 1),
            window_seconds=self._window_seconds,
        )
