from app.llm.base import LLMProvider
from app.schemas.chat import ChatRequest, ChatResponse, Message, Usage


class ChatService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def generate_reply(self, request: ChatRequest) -> ChatResponse:
        completion = await self._provider.complete(request.model, request.messages, request.params)
        return ChatResponse(
            model=request.model,
            message=Message(role="assistant", content=completion.content),
            usage=Usage(
                input_tokens=completion.input_tokens, output_tokens=completion.output_tokens
            ),
        )
