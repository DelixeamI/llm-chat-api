"""Two questions about the context window, answered against a real Ollama.

1. How far is a cheap local estimate (characters / N) from the provider's own token count,
   for Russian and English text?
2. What does Ollama do with a prompt longer than its context window: error, or something else?

Run: python scripts/context_experiment.py
"""

import asyncio
import math

from openai import AsyncOpenAI

from app.config import get_settings

MODEL = "qwen3:8b"

SAMPLES = {
    "ru_short": "Не могу войти в аккаунт после смены пароля.",
    "ru_long": "С карты дважды списали оплату за подписку за сентябрь. " * 20,
    "en_long": "The export to CSV produces broken encoding for Cyrillic characters. " * 20,
    "code": "def handler(request):\n    return {'status': 'ok', 'items': [1, 2, 3]}\n" * 20,
    "mixed": "Ошибка 500 на POST /api/v1/orders при payload > 1MB, see logs. " * 20,
}


async def prompt_tokens(client: AsyncOpenAI, text: str) -> int:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
        max_tokens=1,
        temperature=0,
        reasoning_effort=get_settings().ollama_reasoning_effort,
    )
    assert response.usage is not None
    return response.usage.prompt_tokens


async def main() -> None:
    settings = get_settings()
    async with AsyncOpenAI(
        base_url=settings.ollama_base_url, api_key="ollama", max_retries=0, timeout=300
    ) as client:
        baseline = await prompt_tokens(client, "a")
        print(f"chat template overhead (prompt 'a'): {baseline} tokens\n")

        print(f"{'sample':10s} {'chars':>6s} {'tokens':>6s} {'chars/token':>11s}")
        for name, text in SAMPLES.items():
            tokens = await prompt_tokens(client, text) - baseline + 1
            print(f"{name:10s} {len(text):6d} {tokens:6d} {len(text) / tokens:11.2f}")

        print("\nprompt larger than the context window (4096 in Ollama):")
        unit = "Сервис не работает, клиенты не могут оформить заказ. "
        # Tokens per repeated unit, measured on a prompt that certainly fits
        per_unit = (await prompt_tokens(client, unit * 100) - baseline) / 100
        for repeats in (100, 300, 450, 600, 1200):
            expected = baseline + math.ceil(per_unit * repeats)
            got = await prompt_tokens(client, unit * repeats)
            note = "" if got >= expected - 5 else "  <- cut without any error"
            print(f"  expected {expected:6d} tokens -> provider counted {got:5d}{note}")


if __name__ == "__main__":
    asyncio.run(main())
