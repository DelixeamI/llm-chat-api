from collections.abc import Mapping
from decimal import Decimal

from pydantic import BaseModel, Field

TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)


class ModelPrice(BaseModel):
    """Price per one million tokens, in USD, as providers usually publish it."""

    input_per_million_usd: Decimal = Field(ge=0)
    output_per_million_usd: Decimal = Field(ge=0)


class PriceList:
    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = dict(prices)

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> Decimal | None:
        """None when the model has no price: an unknown cost is not a zero cost."""
        price = self._prices.get(model)
        if price is None:
            return None
        # Input and output are priced separately: generating a token costs the provider
        # far more compute than reading one, so output is usually several times dearer
        return (
            input_tokens * price.input_per_million_usd
            + output_tokens * price.output_per_million_usd
        ) / TOKENS_PER_PRICE_UNIT
