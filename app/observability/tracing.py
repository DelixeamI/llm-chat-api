"""Distributed tracing with OpenTelemetry: where one request spent its time.

A trace is the story of one request; a span is one step in it (a database read, a model call)
with a start, an end and a few attributes; spans nest, so the trace is a tree. For a chat
request that is the useful picture: of 4 seconds, how much was the queue for a model slot, how
much the model, how much the database. A metric would show that the average rose, a log line
would show one step, only a trace shows how the steps add up.

Tracing is off unless configured: the spans below are then no-ops and cost almost nothing.

What goes into attributes: model, counts, durations, outcomes. Never prompts, replies, keys or
any user text, for the same reasons as in the logs (docs/logging.md).
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)

SERVICE_NAME = "llm-chat-api"

# Our own provider rather than the process-wide one: the global can be set once per process,
# which makes tests and reloads fragile. Without one, spans come from the no-op global tracer.
_provider: TracerProvider | None = None


def get_tracer() -> Tracer:
    if _provider is not None:
        return _provider.get_tracer(SERVICE_NAME)
    return trace.get_tracer(SERVICE_NAME)


def use_provider(provider: TracerProvider | None) -> None:
    global _provider
    _provider = provider


def configure_tracing(
    *,
    enabled: bool,
    otlp_endpoint: str | None = None,
    console: bool = False,
    sample_ratio: float = 1.0,
) -> TracerProvider | None:
    """Build the provider: where spans go and how many requests are traced."""
    if not enabled:
        use_provider(None)
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE_NAME}),
        # ParentBased: a request that arrives already inside a sampled trace stays in it, so a
        # trace is never cut in half by two services deciding differently
        sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
    )
    processors: list[SpanProcessor] = []
    if otlp_endpoint:
        # Batched and exported in the background: a slow collector must not slow requests
        processors.append(BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint)))
    if console:
        processors.append(SimpleSpanProcessor(ConsoleSpanExporter()))
    if not processors:
        logger.warning("Tracing is enabled but has no destination (TRACING_OTLP_ENDPOINT)")
    for processor in processors:
        provider.add_span_processor(processor)
    use_provider(provider)
    return provider


def extract_parent(headers: Mapping[str, str]) -> Context:
    """Continue the caller's trace if it sent a W3C `traceparent` header."""
    return extract(headers)


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return f"{context.trace_id:032x}" if context.is_valid else None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """A child span of the current one. Marks itself as failed if the block raises."""
    # The library would otherwise put "ExceptionType: message" into the span status and
    # record the exception with its message: a message can quote user data
    with get_tracer().start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        try:
            yield current
        except BaseException as exc:
            # The type only: the message of an exception can carry user data
            current.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            current.set_attribute("error.type", type(exc).__name__)
            raise


def start_detached(name: str, **attributes: Any) -> Span:
    """A span that is not made current, for code that yields inside it (streams).

    Making a span current stores a token in the context; an async generator can resume in a
    different context than it started in, and detaching the token there fails. A detached span
    avoids the problem: its parent is passed explicitly and it is ended by hand.
    """
    detached = get_tracer().start_span(
        name, context=trace.set_span_in_context(trace.get_current_span())
    )
    for key, value in attributes.items():
        if value is not None:
            detached.set_attribute(key, value)
    return detached


def server_span(name: str, parent: Context, **attributes: Any) -> AbstractContextManager[Span]:
    """The root span of a request, continuing the caller's trace when there is one."""
    return get_tracer().start_as_current_span(
        name,
        context=parent,
        kind=SpanKind.SERVER,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    )
