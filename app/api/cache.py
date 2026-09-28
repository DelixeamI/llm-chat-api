from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_response_cache
from app.schemas.cache import CacheStats
from app.schemas.errors import ErrorResponse
from app.services.cache import ResponseCache

router = APIRouter(prefix="/v1/cache", tags=["cache"])


@router.get("/stats", response_model=CacheStats, responses={503: {"model": ErrorResponse}})
async def cache_stats(
    cache: Annotated[ResponseCache | None, Depends(get_response_cache)],
) -> CacheStats:
    """Hits and misses of the response cache, shared by all processes using this Redis."""
    if cache is None:
        return CacheStats(enabled=False, hits=0, misses=0)
    return await cache.stats()
