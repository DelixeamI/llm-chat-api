"""Turning raw model text into validated application data.

Two separate steps with two separate errors: parsing answers "is this JSON at all", validation
answers "is this JSON the data we need". Syntactically valid JSON is often still unusable.
"""

import json
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError


class StructuredOutputError(Exception):
    """The model answered, but the answer is not usable data."""

    kind: ClassVar[str]


class OutputNotJSONError(StructuredOutputError):
    kind = "invalid_json"

    def __init__(self, reason: str) -> None:
        super().__init__(f"model output is not valid JSON: {reason}")
        self.reason = reason


class OutputSchemaError(StructuredOutputError):
    kind = "schema_mismatch"

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in errors
        )
        super().__init__(f"model output does not match the schema: {problems}")
        self.errors = errors


def parse_output[T: BaseModel](raw: str, schema: type[T]) -> T:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Position, not content: the raw text may contain user data and must stay out of logs
        raise OutputNotJSONError(f"{exc.msg} at line {exc.lineno} column {exc.colno}") from exc

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        # include_input=False keeps the offending values (possibly user data) out of the error
        raise OutputSchemaError(
            [dict(error) for error in exc.errors(include_url=False, include_input=False)]
        ) from exc
