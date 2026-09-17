"""Support for tests that talk to a real PostgreSQL.

The test database is separate from the development one, its schema is built by the same
Alembic migrations as production, and every test starts from empty tables.
"""

import asyncio
from collections.abc import Awaitable, Callable

from alembic import command
from alembic.config import Config
from sqlalchemy import URL, make_url, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import get_settings
from app.db.session import create_engine, create_session_factory

TABLES = ("messages", "conversations")

_prepared = False


def test_database_url() -> URL:
    url = make_url(get_settings().database_url)
    return url.set(database=f"{url.database}_test")


async def _create_database_if_missing(url: URL) -> None:
    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT
    admin = create_async_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as connection:
            exists = await connection.scalar(
                text("select 1 from pg_database where datname = :name"), {"name": url.database}
            )
            if not exists:
                await connection.execute(text(f'create database "{url.database}"'))
    finally:
        await admin.dispose()


def prepare_database() -> URL | None:
    """Create the test database and migrate it. Returns None if PostgreSQL is unreachable."""
    global _prepared
    url = test_database_url()
    if _prepared:
        return url
    try:
        asyncio.run(_create_database_if_missing(url))
    except Exception:
        return None

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url.render_as_string(hide_password=False))
    command.upgrade(config, "head")
    _prepared = True
    return url


async def _truncate(session: AsyncSession) -> None:
    # TRUNCATE is faster than DELETE and resets the tables completely; CASCADE follows
    # the foreign key from messages
    await session.execute(text(f"truncate table {', '.join(TABLES)} cascade"))
    await session.commit()


def run_with_session[T](scenario: Callable[[AsyncSession], Awaitable[T]], url: URL) -> T:
    """Run one scenario against empty tables in its own engine and event loop."""

    async def runner() -> T:
        engine = create_engine(url.render_as_string(hide_password=False))
        factory = create_session_factory(engine)
        try:
            async with factory() as session:
                await _truncate(session)
            async with factory() as session:
                return await scenario(session)
        finally:
            await engine.dispose()

    return asyncio.run(runner())
