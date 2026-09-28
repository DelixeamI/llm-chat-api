"""Keeping requests inside the model's context window before the provider sees them.

Ollama does not reject an oversized prompt: it silently cuts it to half the window and the
model answers from what is left. The check has to happen on our side.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from app.schemas.chat import Message
from app.services.errors import ContextOverflowError

OverflowStrategy = Literal["reject", "truncate_oldest"]

# Measured against qwen3:8b: Russian text and code run 2.7-2.9 characters per token, English
# 5.6. 2.5 overestimates all of them, which is the safe direction for a limit check. The
# familiar "4 characters per token" rule would undercount Russian by about 40%.
# ponytail: a heuristic, not a tokenizer. Overestimation rejects some requests that would fit
# (seen: ~3200 real tokens estimated as 5173). Upgrade path: the model's own tokenizer.
CHARS_PER_TOKEN = 2.5
# Chat template framing around each message (role markers, separators); a one-word prompt
# measured 17 tokens in total
PER_MESSAGE_TOKENS = 8
BASE_TOKENS = 10


@dataclass(frozen=True)
class ContextBudget:
    limits: Mapping[str, int]
    default_limit: int
    strategy: OverflowStrategy = "reject"

    def limit_for(self, model: str) -> int:
        return self.limits.get(model, self.default_limit)

    def estimate(self, messages: list[Message]) -> int:
        """Rough upper bound, no tokenizer: the real count is known only after the call."""
        return BASE_TOKENS + sum(
            PER_MESSAGE_TOKENS + math.ceil(len(m.content) / CHARS_PER_TOKEN) for m in messages
        )

    def check(self, model: str, messages: list[Message], max_tokens: int) -> None:
        # The output budget is reserved up front: input that fits but leaves no room for the
        # answer produces a truncated answer instead of an error
        estimated = self.estimate(messages)
        limit = self.limit_for(model)
        if estimated + max_tokens > limit:
            raise ContextOverflowError(estimated=estimated, reserved=max_tokens, limit=limit)

    def fit(
        self, model: str, messages: list[Message], max_tokens: int
    ) -> tuple[list[Message], int]:
        """Apply the configured strategy. Returns the messages to send and how many were dropped."""
        if self.strategy == "reject":
            self.check(model, messages, max_tokens)
            return messages, 0

        # Drop the oldest dialogue turns first. System messages carry the instructions and the
        # last user message is the question itself: without either the answer is meaningless.
        last_user = max(i for i, m in enumerate(messages) if m.role == "user")
        droppable = [i for i, m in enumerate(messages) if m.role != "system" and i != last_user]
        kept = list(range(len(messages)))
        dropped = 0
        while True:
            candidate = [messages[i] for i in kept]
            try:
                self.check(model, candidate, max_tokens)
            except ContextOverflowError:
                if not droppable:
                    # Even the essentials do not fit: truncating further would change the task
                    raise
                kept.remove(droppable.pop(0))
                dropped += 1
                continue
            return candidate, dropped
