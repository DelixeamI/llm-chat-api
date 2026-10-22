from pydantic import BaseModel, Field, computed_field


class CacheStats(BaseModel):
    enabled: bool
    # Lookups for cacheable requests since Redis started; other requests never reach the cache
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hit_rate(self) -> float | None:
        lookups = self.hits + self.misses
        return round(self.hits / lookups, 4) if lookups else None
