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
