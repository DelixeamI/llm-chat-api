from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request, Response, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.api.dependencies import get_rate_limiter
from app.auth import Authenticator, Client, Scope
from app.observability import metrics
from app.observability.context import client_var
from app.services.errors import AuthenticationError, InsufficientScopeError, RateLimitExceededError
from app.services.ratelimit import RateLimiter, RateLimitResult

# auto_error=False: FastAPI's own failure would be a 403 with a generic body. A missing key is
# an authentication problem (401) in our error format, so the dependency raises it itself.
# Declaring the scheme still puts the "Authorize" button into Swagger UI.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# The same key may also come as `Authorization: Bearer <key>`: a Prometheus scraper can send
# that header but not a custom one
bearer = HTTPBearer(auto_error=False)


def get_authenticator(request: Request) -> Authenticator:
    authenticator: Authenticator = request.app.state.authenticator
    return authenticator


async def current_client(
    api_key: Annotated[str | None, Security(api_key_header)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
    authenticator: Annotated[Authenticator, Depends(get_authenticator)],
) -> Client:
    """Who is calling. Unknown or missing key: 401."""
    client = authenticator.authenticate(
        api_key or (credentials.credentials if credentials else None)
    )
    if client is None:
        metrics.AUTH_FAILURES.labels("invalid_key").inc()
        raise AuthenticationError()
    # From here on every log line of this request names the client. The dependency is async
    # on purpose: a sync one would run in a worker thread with a copy of the context, and the
    # value set here would be gone by the time the request line is written.
    client_var.set(client.name)
    return client


def require_scope(*scopes: Scope) -> Callable[..., Client]:
    """What they may do. A known client with none of the given scopes: 403."""

    def check(client: Annotated[Client, Depends(current_client)]) -> Client:
        if not any(client.has(scope) for scope in scopes):
            metrics.AUTH_FAILURES.labels("insufficient_scope").inc()
            raise InsufficientScopeError(client=client.name, scope=" or ".join(scopes))
        return client

    return check


ChatClient = Annotated[Client, Depends(require_scope("chat"))]
# Reading conversations: the owner (chat) or an operator (admin)
ConversationClient = Annotated[Client, Depends(require_scope("chat", "admin"))]
AdminClient = Annotated[Client, Depends(require_scope("admin"))]


async def enforce_rate_limit(
    client: ChatClient,
    response: Response,
    limiter: Annotated[RateLimiter | None, Depends(get_rate_limiter)],
) -> RateLimitResult | None:
    """Count the request against the client's window; 429 when the window is full.

    Runs after authentication, so only identified clients are counted: a flood of requests
    with wrong keys cannot use up a real client's allowance.
    """
    if limiter is None:
        return None
    result = await limiter.check(client.name)
    if result is None:
        return None
    if not result.admitted:
        metrics.RATE_LIMITED.inc()
        raise RateLimitExceededError(
            client=client.name,
            headers=result.headers(),
            limit=result.limit,
            window=result.window_seconds,
        )
    response.headers.update(result.headers())
    return result


RateLimited = Annotated[RateLimitResult | None, Depends(enforce_rate_limit)]
