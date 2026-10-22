"""Facts about the current request that every log line should carry.

A context variable follows the request through awaits and into the tasks it starts, so code
deep inside a service can log with the request's identity without receiving it as an argument.
"""

from contextvars import ContextVar

# Correlation ID: one value for everything that happens because of one request. With it, the
# scattered lines of a request (route, cache, model call, database) can be pulled out of a log
# that mixes thousands of requests, and a client can quote it when reporting a problem.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
# Who the authenticated caller is; set once authentication has run
client_var: ContextVar[str | None] = ContextVar("client", default=None)
