"""Prompt-only JSON vs JSON mode vs Structured Output on the same naive prompt.

The prompt deliberately does not list allowed values: the point is to see what each
mechanism guarantees on its own, not what a carefully worded prompt can coax out.

Run: python scripts/structured_output_experiment.py
"""

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, omit

from app.config import get_settings
from app.llm.structured import StructuredOutputError, parse_output
from app.schemas.analysis import SupportAnalysis
from scripts.json_prompt_experiment import NAIVE, TICKETS

MODEL = "qwen3:8b"
RESULTS = Path("docs/experiments/structured_output_comparison.jsonl")
SCHEMA = SupportAnalysis.model_json_schema()

MODES: dict[str, Any] = {
    "prompt_only": omit,
    "json_mode": {"type": "json_object"},
    "json_schema": {
        "type": "json_schema",
        "json_schema": {"name": "support_analysis", "schema": SCHEMA, "strict": True},
    },
}


def outcome(raw: str) -> str:
    try:
        parse_output(raw, SupportAnalysis)
    except StructuredOutputError as exc:
        return exc.kind
    return "ok"


async def main() -> None:
    settings = get_settings()
    records: list[dict[str, Any]] = []

    async with AsyncOpenAI(
        base_url=settings.ollama_base_url, api_key="ollama", max_retries=0, timeout=120
    ) as client:

        async def ask(mode: str, ticket: str, max_tokens: int) -> str:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": NAIVE.format(ticket=ticket)}],
                temperature=0.7,
                max_tokens=max_tokens,
                reasoning_effort=settings.ollama_reasoning_effort,
                response_format=MODES[mode],
            )
            return response.choices[0].message.content or ""

        for mode in MODES:
            for index, ticket in enumerate(TICKETS):
                raw = await ask(mode, ticket, max_tokens=300)
                records.append({"mode": mode, "ticket": index, "raw": raw, "outcome": outcome(raw)})

        # The schema constrains which tokens may come next, not how many: a tight token
        # limit cuts the object in half, and nothing in the grammar can prevent that
        for index, ticket in enumerate(TICKETS[:4]):
            raw = await ask("json_schema", ticket, max_tokens=12)
            records.append(
                {
                    "mode": "json_schema_max_tokens_12",
                    "ticket": index,
                    "raw": raw,
                    "outcome": outcome(raw),
                }
            )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    for mode in [*MODES, "json_schema_max_tokens_12"]:
        outcomes = Counter(r["outcome"] for r in records if r["mode"] == mode)
        print(f"[{mode}] {dict(outcomes)}")
    categories = Counter(
        json.loads(r["raw"]).get("category") for r in records if r["mode"] == "json_mode"
    )
    print("json_mode categories:", dict(categories))
    print(f"raw responses: {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
