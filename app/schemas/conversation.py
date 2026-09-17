import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)


class ConversationResponse(BaseModel):
    # from_attributes lets FastAPI build this straight from an ORM row
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: Literal["system", "user", "assistant"]
    content: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime
