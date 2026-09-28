from decimal import Decimal

import pytest

from app.services.pricing import ModelPrice, PriceList

PRICES = PriceList(
    {
        "cheap": ModelPrice(
            input_per_million_usd=Decimal("0.15"), output_per_million_usd=Decimal("0.60")
        )
    }
)


def test_cost_is_priced_separately_for_input_and_output() -> None:
    # 2000 * 0.15 / 1e6 + 500 * 0.60 / 1e6 = 0.0003 + 0.0003
    assert PRICES.cost_usd("cheap", 2000, 500) == Decimal("0.0006")


def test_output_tokens_cost_more_than_input_tokens() -> None:
    reading = PRICES.cost_usd("cheap", 1000, 0)
    writing = PRICES.cost_usd("cheap", 0, 1000)

    assert reading is not None and writing is not None
    assert writing == reading * 4


def test_unknown_model_has_no_cost_rather_than_zero() -> None:
    assert PRICES.cost_usd("never-priced", 1000, 1000) is None


def test_cost_is_exact_decimal_not_float() -> None:
    cost = PRICES.cost_usd("cheap", 1, 1)

    assert isinstance(cost, Decimal)
    # With floats 0.15e-6 + 0.60e-6 would not be exactly 0.00000075
    assert cost == Decimal("0.00000075")


def test_negative_price_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelPrice(input_per_million_usd=Decimal("-1"), output_per_million_usd=Decimal("1"))
