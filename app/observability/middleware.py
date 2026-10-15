import logging
import re
import time
import uuid

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability import metrics
from app.observability.context import client_var, request_id_var
from app.schemas.errors import ErrorResponse

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "X-Request-ID"
# A caller may supply its own ID, so that one ID can follow a request across several services.
# The value ends up in logs and response headers, so only a short, plain alphabet is accepted:
# anything else could forge log lines or smuggle in header content.
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


def _request_id(scope: Scope) -> str:
    for name, value in scope["headers"]:
        if name == b"x-request-id":
            supplied: str = value.decode("latin-1")
            if _VALID_ID.fullmatch(supplied):
                return supplied
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Gives each request an ID, echoes it back, and logs one line when the request ends.

    A plain ASGI middleware rather than BaseHTTPMiddleware: the latter wraps the response in
    an extra task and interferes with streaming and with client disconnects.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)
        request_id_token = request_id_var.set(request_id)
        client_token = client_var.set(None)
        started = time.perf_counter()
        status = 500
        response_started = False
        metrics.HTTP_IN_FLIGHT.inc()

        async def send_with_id(message: Message) -> None:
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            # An error nothing else handled. The outermost layer (Starlette's) would answer
            # with a bare 500 that carries no request ID, which is exactly the response a user
            # reports. Answer here instead, with the ID, and keep the traceback in the log.
            logger.exception("Unhandled error", extra={"event": "unhandled_error"})
            if response_started:
                raise
            body = ErrorResponse(
                error="internal_error", detail=f"Unexpected error, request id {request_id}"
            )
            await JSONResponse(
                status_code=500, content=body.model_dump(), headers={REQUEST_ID_HEADER: request_id}
            )(scope, receive, send_with_id)
        finally:
            route = scope.get("route")
            # The route template, and "unmatched" for unknown paths: a label per real URL
            # would create a new time series for every identifier and every scanner probe
            template = getattr(route, "path", "unmatched")
            elapsed = time.perf_counter() - started
            metrics.HTTP_IN_FLIGHT.dec()
            metrics.HTTP_REQUESTS.labels(scope["method"], template, str(status)).inc()
            metrics.HTTP_DURATION.labels(scope["method"], template).observe(elapsed)
            logger.log(
                logging.ERROR if status >= 500 else logging.INFO,
                "request",
                extra={
                    "event": "request",
                    "method": scope["method"],
                    # The route template ("/v1/conversations/{conversation_id}"), not the real
                    # path: real paths hold identifiers and make every URL a separate series
                    "path": getattr(route, "path", "unmatched"),
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            client_var.reset(client_token)
            request_id_var.reset(request_id_token)
