import uuid


class ConversationNotFoundError(Exception):
    """Raised when a request references a conversation that does not exist."""

    def __init__(self, conversation_id: uuid.UUID) -> None:
        super().__init__(f"conversation {conversation_id} not found")
        self.conversation_id = conversation_id
