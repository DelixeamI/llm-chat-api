# llm-chat-api

Production-oriented asynchronous LLM backend: validates chat requests, calls LLM providers with
timeouts and retries, persists conversations and token usage, calculates cost, streams responses,
and exposes observability. Built as a learning project for Middle AI / LLM Engineer skills.

## Status

Day 1 — project bootstrap. No endpoints yet.

## Stack

Python 3.12+, FastAPI, Pydantic v2. PostgreSQL, Redis, Docker and observability arrive in later
stages.

## Project structure

```
app/      application code
tests/    pytest suite
```

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in local values. `.env` is never committed.

## Roadmap

Six stages: FastAPI + Pydantic foundations, async LLM integration, PostgreSQL persistence,
structured outputs and cost accounting, Redis and streaming, observability and CI/CD.
