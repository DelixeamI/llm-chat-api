from redis.asyncio import BlockingConnectionPool, Redis

# Every key this service writes starts with this prefix, so its data can be told apart,
# listed and cleared without touching anything else stored in the same Redis
KEY_PREFIX = "llm-chat"


def make_key(*parts: str) -> str:
    """Namespaced key: make_key("cache", "v1", digest) -> "llm-chat:cache:v1:<digest>"."""
    return ":".join((KEY_PREFIX, *parts))


def create_redis(url: str, *, max_connections: int, timeout_seconds: float) -> Redis:
    """One client per process; it owns the connection pool, so it must outlive requests.

    Nothing connects here: the pool opens connections on first use, so the application
    starts even while Redis is down, and each feature decides how to degrade.
    """
    pool = BlockingConnectionPool.from_url(
        url,
        # A blocking pool makes a caller wait for a free connection; the default pool fails
        # at once with "Too many connections", turning a short burst into errors
        max_connections=max_connections,
        # How long to wait for a free connection from the pool
        timeout=timeout_seconds,
        # Redis answers in well under a millisecond; a slow or unreachable Redis must fail
        # fast rather than stall the request that only wanted a cache lookup
        socket_connect_timeout=timeout_seconds,
        socket_timeout=timeout_seconds,
    )
    # from_pool hands the pool to the client, so closing the client closes the connections.
    # Redis(connection_pool=pool) would not: aclose() at shutdown would leave them open.
    return Redis.from_pool(pool)
