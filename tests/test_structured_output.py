import json

import pytest

from app.llm.structured import OutputNotJSONError, OutputSchemaError, parse_output
from app.schemas.analysis import SupportAnalysis

VALID = {
    "category": "billing",
    "priority": "high",
    "summary": "Двойное списание",
    "confidence": 0.9,
}


def raw(**overrides: object) -> str:
    return json.dumps(VALID | overrides, ensure_ascii=False)


def test_valid_output_is_parsed() -> None:
    analysis = parse_output(raw(), SupportAnalysis)

    assert analysis == SupportAnalysis.model_validate(VALID)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(f"```json\n{raw()}\n```", id="markdown-fence"),
        pytest.param(f"Вот результат:\n{raw()}", id="text-before-json"),
        pytest.param(f"{raw()}\nНадеюсь, это поможет!", id="text-after-json"),
        pytest.param('{"category": "billing",', id="truncated"),
        pytest.param("", id="empty"),
    ],
)
def test_non_json_output_is_a_parse_error(text: str) -> None:
    with pytest.raises(OutputNotJSONError):
        parse_output(text, SupportAnalysis)


@pytest.mark.parametrize(
    ("text", "field", "error_type"),
    [
        pytest.param(
            json.dumps({k: v for k, v in VALID.items() if k != "priority"}),
            "priority",
            "missing",
            id="missing-field",
        ),
        pytest.param(raw(confidence="high"), "confidence", "float_parsing", id="wrong-type"),
        pytest.param(raw(confidence=1.7), "confidence", "less_than_equal", id="out-of-range"),
        pytest.param(raw(category="Authentication"), "category", "literal_error", id="enum"),
        pytest.param(raw(priority="High"), "priority", "literal_error", id="enum-case"),
        pytest.param(raw(reason="потому что"), "reason", "extra_forbidden", id="extra-field"),
        pytest.param(raw(summary=""), "summary", "string_too_short", id="empty-summary"),
    ],
)
def test_valid_json_with_wrong_shape_is_a_schema_error(
    text: str, field: str, error_type: str
) -> None:
    with pytest.raises(OutputSchemaError) as exc_info:
        parse_output(text, SupportAnalysis)

    error = exc_info.value.errors[0]
    assert error["loc"] == (field,)
    assert error["type"] == error_type


def test_json_array_is_a_schema_error_not_a_parse_error() -> None:
    with pytest.raises(OutputSchemaError) as exc_info:
        parse_output("[1, 2, 3]", SupportAnalysis)

    assert exc_info.value.errors[0]["type"] == "model_type"


def test_schema_error_does_not_carry_offending_values() -> None:
    secret = "номер карты 4111 1111 1111 1111"

    with pytest.raises(OutputSchemaError) as exc_info:
        parse_output(raw(category=secret), SupportAnalysis)

    assert secret not in str(exc_info.value)
    assert all("input" not in error for error in exc_info.value.errors)


def test_numeric_string_confidence_is_coerced() -> None:
    # Lax mode on purpose: "0.9" carries no ambiguity, rejecting it would waste a generation
    assert parse_output(raw(confidence="0.9"), SupportAnalysis).confidence == 0.9
