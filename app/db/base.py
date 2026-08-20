from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def init_database(database_url: str) -> Engine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    engine_options = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        engine_options["connect_args"] = {"check_same_thread": False}
    elif _serverless_pooling_enabled():
        engine_options["poolclass"] = NullPool
    _engine = create_engine(database_url, **engine_options)
    if database_url.startswith("sqlite"):
        event.listen(_engine, "connect", _enable_sqlite_foreign_keys)

    from app.models import tables  # noqa: F401

    Base.metadata.create_all(_engine)
    if database_url.startswith("sqlite"):
        _migrate_sqlite_schema(_engine)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _migrate_sqlite_schema(engine: Engine) -> None:
    participant_columns = {
        column["name"] for column in inspect(engine).get_columns("participants")
    }
    if "preferred_face_region" not in participant_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE participants "
                    "ADD COLUMN preferred_face_region VARCHAR(32)"
                )
            )
    if "preferred_age_appearance" not in participant_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE participants "
                    "ADD COLUMN preferred_age_appearance VARCHAR(32)"
                )
            )
    _add_sqlite_column_if_missing(
        engine,
        "participants",
        "consent_version",
        "ALTER TABLE participants ADD COLUMN consent_version VARCHAR(32)",
    )
    columns = {
        column["name"] for column in inspect(engine).get_columns("experiment_blocks")
    }
    if "strategy_parameters_json" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE experiment_blocks "
                    "ADD COLUMN strategy_parameters_json TEXT NOT NULL DEFAULT '{}'"
                )
            )
    _add_sqlite_column_if_missing(
        engine,
        "experiment_blocks",
        "session_id",
        "ALTER TABLE experiment_blocks ADD COLUMN session_id VARCHAR(36)",
    )
    _add_sqlite_column_if_missing(
        engine,
        "experiment_blocks",
        "mu_data",
        "ALTER TABLE experiment_blocks ADD COLUMN mu_data BLOB",
    )
    _add_sqlite_column_if_missing(
        engine,
        "experiment_blocks",
        "strategy_state_data",
        "ALTER TABLE experiment_blocks ADD COLUMN strategy_state_data BLOB",
    )
    _add_sqlite_column_if_missing(
        engine,
        "latent_images",
        "latent_data",
        "ALTER TABLE latent_images ADD COLUMN latent_data BLOB",
    )
    _add_sqlite_column_if_missing(
        engine,
        "latent_images",
        "image_data",
        "ALTER TABLE latent_images ADD COLUMN image_data BLOB",
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_selection_block_round "
                "ON selections(block_id, round_id)"
            )
        )


def _add_sqlite_column_if_missing(
    engine: Engine,
    table_name: str,
    column_name: str,
    statement: str,
) -> None:
    columns = {
        column["name"] for column in inspect(engine).get_columns(table_name)
    }
    if column_name in columns:
        return
    with engine.begin() as connection:
        connection.execute(text(statement))


def _serverless_pooling_enabled() -> bool:
    value = os.getenv("IDEAL_DB_POOL", "").lower()
    return value == "serverless" or os.getenv("VERCEL") == "1"


def get_session() -> Session:
    if _session_factory is None:
        raise RuntimeError("Database has not been initialized")
    return _session_factory()


def get_db() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def reset_database_for_tests(database_url: str) -> Engine:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    return init_database(database_url)
