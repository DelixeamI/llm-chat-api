"""Prometheus metrics: numbers about the service, scraped and charted over time.

Logs say what happened to one request; metrics say how the whole service is doing: how many
requests, how many failed, how long the slow ones take. A log line is an event, a metric is
an aggregate, so metrics stay cheap however much traffic there is.

Three kinds are used:

- Counter: only goes up (requests, errors, tokens). Charts show its rate of change.
- Gauge: goes up and down (calls in flight right now).
- Histogram: counts observations into buckets (latency). Percentiles come from the buckets.

Labels multiply the number of time series, so each label here takes few, fixed values: a
route template instead of a path, a model name, a status. Never an identifier, a prompt or a
client-supplied string: one series per value would grow without bound.

The registry is per process. Under several uvicorn workers each would report only its own
share; prometheus_client's multiprocess mode exists for that and is not used here.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Seconds. HTTP: most requests are fast (a lookup) or as slow as a model call.
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
# A model call takes from a fraction of a second to minutes; buckets cover the whole range so
# that a slow tail shows up as a bucket, not as a clipped average.
LLM_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180)
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests handled", ["method", "path", "status"]
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration, from first byte received to last byte sent",
    ["method", "path"],
    buckets=HTTP_BUCKETS,
)
HTTP_IN_FLIGHT = Gauge("http_requests_in_flight", "HTTP requests being handled right now")

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Model interactions by outcome",
    ["model", "endpoint", "status"],
)
LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "Model call duration, retries included",
    ["model", "endpoint"],
    buckets=LLM_BUCKETS,
)
LLM_TTFT = Histogram(
    "llm_ttft_seconds",
    "Streaming: time to the first piece of text",
    ["model"],
    buckets=TTFT_BUCKETS,
)
LLM_ERRORS = Counter("llm_errors_total", "Failed model interactions", ["model", "kind"])
LLM_RETRIES = Counter("llm_retries_total", "Retried model attempts", ["model"])
LLM_TOKENS = Counter(
    "llm_tokens_total", "Tokens used, as reported by the provider", ["model", "direction"]
)
LLM_COST = Counter(
    "llm_cost_usd_total", "Cost of model calls in USD, at the prices configured", ["model"]
)
LLM_IN_FLIGHT = Gauge("llm_calls_in_flight", "Model calls running right now")

STRUCTURED_FAILURES = Counter(
    "structured_output_failures_total",
    "Structured answers rejected by validation; final=true when the request gave up",
    ["kind", "final"],
)

DEPENDENCY_UP = Gauge(
    "dependency_up", "1 if the last readiness check of a dependency passed, else 0", ["dependency"]
)

CACHE_HITS = Counter("cache_hits_total", "Replies served from the response cache")
CACHE_MISSES = Counter("cache_misses_total", "Cacheable requests that found no stored reply")
CACHE_ERRORS = Counter("cache_errors_total", "Cache operations skipped because Redis failed")

RATE_LIMITED = Counter("rate_limit_rejections_total", "Requests refused by the rate limit")
IDEMPOTENT_REPLAYS = Counter(
    "idempotent_replays_total", "Responses replayed for a repeated Idempotency-Key"
)
AUTH_FAILURES = Counter(
    "auth_failures_total", "Requests refused by authentication or authorization", ["reason"]
)


_known_models: frozenset[str] = frozenset()


def allow_models(models: set[str] | frozenset[str]) -> None:
    """Set the model names that may appear as a label value."""
    global _known_models
    _known_models = frozenset(models)


def model_label(model: str) -> str:
    """The model name as a label, or "other" for a name nobody configured.

    The model comes from the request body, i.e. from the client. Used as a label verbatim it
    would let anyone create a new time series per request by inventing model names, and the
    metrics store would grow until it fell over.
    """
    return model if model in _known_models else "other"


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
