from fastapi import APIRouter, Response

from app.api.security import AdminClient
from app.observability import metrics
from app.schemas.errors import ErrorResponse

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_class=Response,
    responses={
        200: {"content": {"text/plain": {}}, "description": "Prometheus text exposition format"},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_metrics(_: AdminClient) -> Response:
    """Service and model metrics for Prometheus. Needs the `admin` scope.

    Numbers about traffic, errors and spending are not for everyone, so the endpoint is not
    public like /health. The key may be sent as X-API-Key or as a Bearer token.
    """
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
