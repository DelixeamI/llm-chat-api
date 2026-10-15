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


class AuthenticationError(Exception):
    """The API key is missing or unknown: we do not know who is calling."""


class InsufficientScopeError(Exception):
    """The caller is known but lacks the scope this endpoint needs."""

    def __init__(self, *, client: str, scope: str) -> None:
        super().__init__(f"this endpoint needs the '{scope}' scope")
        self.client = client
        self.scope = scope
