import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, Message


def test_message_content_is_stripped() -> None:
    assert Message(role="user", content="  hi  ").content == "hi"


def test_generation_params_defaults_are_applied() -> None:
    request = ChatRequest(model="llama3", messages=[Message(role="user", content="hi")])

    assert request.params.temperature == 0.7
    assert request.params.max_tokens == 1024


def test_request_without_user_message_raises() -> None:
    with pytest.raises(ValidationError, match="role 'user'"):
        ChatRequest(model="llama3", messages=[Message(role="system", content="be nice")])
