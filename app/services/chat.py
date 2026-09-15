from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage


async def generate_reply(request: ChatRequest) -> ChatResponse:
    last_user_message = next(m for m in reversed(request.messages) if m.role == "user")

    # ponytail: заглушка вместо вызова модели; настоящий провайдер приходит на дне 8,
    # реальный подсчёт токенов — на дне 26
    return ChatResponse(
        model=request.model,
        message=Message(
            role="assistant", content=f"(заглушка) вы написали: {last_user_message.content}"
        ),
        usage=Usage(input_tokens=0, output_tokens=0),
    )
