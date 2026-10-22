import uuid


class ConversationNotFoundError(Exception):
    """Raised when a request references a conversation that does not exist."""

    def __init__(self, conversation_id: uuid.UUID) -> None:
        super().__init__(f"conversation {conversation_id} not found")
        self.conversation_id = conversation_id


class ContextOverflowError(Exception):
    """The request would not fit into the model's context window."""

    def __init__(self, *, estimated: int, reserved: int, limit: int) -> None:
        super().__init__(
            f"about {estimated} input tokens plus {reserved} reserved for the answer "
            f"exceed the model context window of {limit} tokens"
        )
        self.estimated = estimated
        self.reserved = reserved
        self.limit = limit


class RateLimitExceededError(Exception):
    """The client sent more requests than its window allows."""

    def __init__(self, *, client: str, headers: dict[str, str], limit: int, window: int) -> None:
        retry = headers["Retry-After"]
        super().__init__(f"limit of {limit} requests per {window} s exceeded, retry in {retry} s")
        self.client = client
        self.headers = headers


class IdempotencyInProgressError(Exception):
    """A request with this Idempotency-Key is still being processed."""

    def __init__(self) -> None:
        super().__init__("a request with this Idempotency-Key is still running")


class IdempotencyKeyReuseError(Exception):
    """The Idempotency-Key was already used for a different request."""

    def __init__(self) -> None:
        super().__init__("this Idempotency-Key was already used with a different request")


class IdempotencyNotSupportedError(Exception):
    """Idempotency-Key was sent to an endpoint that cannot honour it."""

    def __init__(self) -> None:
        super().__init__("Idempotency-Key is not supported on this endpoint")


class AuthenticationError(Exception):
    """The API key is missing or unknown: we do not know who is calling."""


class InsufficientScopeError(Exception):
    """The caller is known but lacks the scope this endpoint needs."""

    def __init__(self, *, client: str, scope: str) -> None:
        super().__init__(f"this endpoint needs the '{scope}' scope")
        self.client = client
        self.scope = scope
