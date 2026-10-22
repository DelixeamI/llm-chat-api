"""Migrations run from the command line and from inside other programs."""

import logging

from alembic import command
from alembic.config import Config

from app.observability.logs import configure_logging


def test_running_migrations_does_not_switch_off_application_loggers() -> None:
    configure_logging("INFO", "json")
    logger = logging.getLogger("app.request")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", "postgresql+asyncpg://u:p@localhost/none")

    # Offline mode renders the SQL and needs no database, but runs the same env.py as a real
    # `alembic upgrade head` from the command line, including its logging setup
    command.upgrade(config, "head", sql=True)

    assert logger.disabled is False
    assert logging.getLogger("app.llm").disabled is False


def test_a_program_running_migrations_keeps_its_own_logging() -> None:
    configure_logging("INFO", "json")
    handlers = list(logging.getLogger().handlers)
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", "postgresql+asyncpg://u:p@localhost/none")
    config.attributes["configure_logger"] = False

    command.upgrade(config, "head", sql=True)

    assert logging.getLogger().handlers == handlers
