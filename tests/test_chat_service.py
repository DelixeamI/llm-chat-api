import asyncio

from app.llm.base import Completion
from app.schemas.chat import ChatRequest, GenerationParams, Message
from tests.fakes import ServiceFactory

REQUEST = ChatRequest(model="fake", messages=[Message(role="user", content="hi")])


class InFlightCounter:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(
        self, model: str, messages: list[Message], params: GenerationParams
    ) -> Completion:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.in_flight -= 1
        return Completion(content="ok", input_tokens=1, output_tokens=1)


def test_concurrent_provider_calls_never_exceed_limit(make_service: ServiceFactory) -> None:
    provider = InFlightCounter()
    service = make_service(provider, max_concurrency=3)

    async def run() -> None:
        await asyncio.gather(*(service.generate_reply(REQUEST) for _ in range(20)))

    asyncio.run(run())

    assert provider.max_in_flight == 3
