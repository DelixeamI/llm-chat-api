"""Readiness: can this instance serve traffic right now?

Liveness and readiness answer different questions and trigger different reactions:

- Liveness (`/health`): is the process alive? If not, restart it. It checks nothing outside the
  process, on purpose: a restart cannot fix a dead database, so restarting every instance
  because the database is down would only add chaos.
- Readiness (`/ready`): are the things this instance needs reachable? If not, stop sending it
  traffic, but leave it running: it may recover on its own.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from app.observability import metrics

logger = logging.getLogger(__name__)

Probe = Callable[[], Awaitable[object]]
Overall = Literal["ready", "degraded", "not_ready"]


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    required: bool
    latency_ms: float


@dataclass(frozen=True)
class ReadinessReport:
    status: Overall
    checks: Mapping[str, CheckResult]

    @property
    def serving(self) -> bool:
        return self.status != "not_ready"


class ReadinessChecker:
    def __init__(
        self,
        probes: Mapping[str, Probe],
        *,
        required: frozenset[str],
        timeout_seconds: float,
        cache_seconds: float,
    ) -> None:
        self._probes = dict(probes)
        self._required = required
        self._timeout_seconds = timeout_seconds
        # Orchestrators probe every few seconds, from several places. Without a cache every
        # probe would reach the database and Redis: the health check itself would become load,
        # and a flood of probes could take down the very dependency it asks about.
        self._cache_seconds = cache_seconds
        self._lock = asyncio.Lock()
        self._cached: tuple[float, ReadinessReport] | None = None

    async def check(self) -> ReadinessReport:
        # One check at a time: probes arriving together wait for it and share its result
        async with self._lock:
            now = time.monotonic()
            if self._cached is not None and now < self._cached[0]:
                return self._cached[1]
            report = await self._run()
            self._cached = (time.monotonic() + self._cache_seconds, report)
            return report

    async def _run(self) -> ReadinessReport:
        names = list(self._probes)
        results = await asyncio.gather(*(self._probe(name) for name in names))
        checks = dict(zip(names, results, strict=True))
        for name, result in checks.items():
            metrics.DEPENDENCY_UP.labels(name).set(1 if result.ok else 0)
        if any(not r.ok and r.required for r in checks.values()):
            status: Overall = "not_ready"
        elif any(not r.ok for r in checks.values()):
            status = "degraded"
        else:
            status = "ready"
        return ReadinessReport(status=status, checks=checks)

    async def _probe(self, name: str) -> CheckResult:
        started = time.perf_counter()
        ok = False
        try:
            # Bounded: a dependency that hangs must produce "not ready" in a second, not a probe
            # that hangs until the orchestrator gives up and kills the instance
            async with asyncio.timeout(self._timeout_seconds):
                await self._probes[name]()
            ok = True
        except TimeoutError:
            logger.warning("Readiness check timed out: dependency=%s", name)
        except Exception as exc:
            # The type only, never the text: it can hold a connection string
            logger.warning(
                "Readiness check failed: dependency=%s error=%s", name, type(exc).__name__
            )
        return CheckResult(
            ok=ok,
            required=name in self._required,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
