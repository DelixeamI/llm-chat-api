import time


class Cooldown:
    """A pause after a failure, so an outage costs one error instead of one per request.

    Features that merely use Redis (cache, rate limit) must not turn its outage into extra
    latency: without a pause every request would wait for the full timeout and log a warning.
    """

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._until = 0.0

    @property
    def seconds(self) -> float:
        return self._seconds

    def active(self) -> bool:
        return time.monotonic() < self._until

    def trip(self) -> None:
        self._until = time.monotonic() + self._seconds
