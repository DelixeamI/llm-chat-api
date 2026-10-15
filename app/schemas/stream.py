"""Events of POST /v1/chat/stream, delivered as Server-Sent Events.

The HTTP status is fixed by the first byte of the body, so once text has started to flow a
failure can no longer change it. Failures before the first piece are ordinary HTTP errors;
failures after it arrive as an `error` event.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.chat import Usage


class StreamDelta(BaseModel):
    event: Literal["delta"] = "delta"
    # A piece of text as the provider produced it: not necessarily one token or one word
    content: str


class StreamDone(BaseModel):
    event: Literal["done"] = "done"
    conversation_id: uuid.UUID
    model: str
    usage: Usage
    finish_reason: str | None
    dropped_messages: int = 0
    cached: bool = False
    # Request start to the first piece of text: what the user perceives as "it started"
    ttft_ms: float = Field(ge=0)
    # Request start to the last piece: what the whole generation took
    duration_ms: float = Field(ge=0)


class StreamError(BaseModel):
    event: Literal["error"] = "error"
    error: str
    detail: str | None = None


StreamEvent = StreamDelta | StreamDone | StreamError
