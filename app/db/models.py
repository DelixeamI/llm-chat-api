import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    # Random ids instead of a counter: sequential ids in a public API leak volume and
    # let clients walk over other people's conversations
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.seq"
    )


class Message(Base):
    __tablename__ = "messages"
    # Messages are always read as "one conversation, in order", so the index covers
    # both the filter and the ordering
    __table_args__ = (Index("ix_messages_conversation_seq", "conversation_id", "seq"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Insertion order, assigned by the database. created_at cannot order messages:
    # now() returns the transaction start time, so both messages of one exchange share
    # it, and the random UUID is no tie-breaker
    seq: Mapped[int] = mapped_column(BigInteger, Identity(), unique=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    # Which model produced the message; null for user messages
    model: Mapped[str | None] = mapped_column(String(128))
    # Token usage as reported by the provider; null for user messages
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class UsageLog(Base):
    """Accounting ledger: one row per request that consumed model tokens.

    Unlike token columns on messages, this covers every paid call, including structured
    analysis and generations that were rejected and never became a message.
    """

    __tablename__ = "usage_logs"
    __table_args__ = (Index("ix_usage_logs_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )
    endpoint: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    # SET NULL, not CASCADE: deleting a conversation must not erase what it cost
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL")
    )
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    # Computed by the database, so it can never disagree with its two parts
    total_tokens: Mapped[int] = mapped_column(
        Integer, Computed("input_tokens + output_tokens", persisted=True)
    )
    attempts: Mapped[int] = mapped_column(Integer, server_default="1")
    status: Mapped[str] = mapped_column(String(16))
