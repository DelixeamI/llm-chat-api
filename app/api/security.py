from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from app.auth import Authenticator, Client, Scope
from app.services.errors import AuthenticationError, InsufficientScopeError

# auto_error=False: FastAPI's own failure would be a 403 with a generic body. A missing key is
# an authentication problem (401) in our error format, so the dependency raises it itself.
# Declaring the scheme still puts the "Authorize" button into Swagger UI.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_authenticator(request: Request) -> Authenticator:
    authenticator: Authenticator = request.app.state.authenticator
    return authenticator


def current_client(
    api_key: Annotated[str | None, Security(api_key_header)],
    authenticator: Annotated[Authenticator, Depends(get_authenticator)],
) -> Client:
    """Who is calling. Unknown or missing key: 401."""
    client = authenticator.authenticate(api_key)
    if client is None:
        raise AuthenticationError()
    return client


def require_scope(*scopes: Scope) -> Callable[..., Client]:
    """What they may do. A known client with none of the given scopes: 403."""

    def check(client: Annotated[Client, Depends(current_client)]) -> Client:
        if not any(client.has(scope) for scope in scopes):
            raise InsufficientScopeError(client=client.name, scope=" or ".join(scopes))
        return client

    return check


ChatClient = Annotated[Client, Depends(require_scope("chat"))]
# Reading conversations: the owner (chat) or an operator (admin)
ConversationClient = Annotated[Client, Depends(require_scope("chat", "admin"))]
AdminClient = Annotated[Client, Depends(require_scope("admin"))]
