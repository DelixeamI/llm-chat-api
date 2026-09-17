from fastapi import Request

from app.services.chat import ChatService


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service
