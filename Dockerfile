# syntax=docker/dockerfile:1

# ---- builder: dependencies only --------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
# Only the lock file goes in before the install. The layer is rebuilt when the pinned
# dependencies change, not on every edit of the code: the slow step stays cached.
COPY requirements.lock ./
RUN pip install -r requirements.lock

# ---- runtime: the venv and the code, nothing else ---------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# A fixed unprivileged user: a flaw in the application must not hand over root in the container
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app

# No compiler, no pip cache, no build tools in the final image: it holds what runs, and the
# builder stage with everything else is discarded
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Code changes most often, so it comes last and invalidates only the layers below it
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app tools ./tools
COPY --chown=app:app app ./app

USER app
EXPOSE 8000

# Liveness only (/health touches no dependency). Python is already in the image; curl is not.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# Exec form, so uvicorn is PID 1 and receives SIGTERM from `docker stop` directly: through a
# shell it would never see the signal and would be killed after the grace period instead.
# On SIGTERM uvicorn stops accepting connections, lets requests in flight finish for up to 30 s
# and then runs the app shutdown (closing the model client, the database pool and Redis).
# --no-access-log: the application writes its own request line with the request ID.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--timeout-graceful-shutdown", "30"]
