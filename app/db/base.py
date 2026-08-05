from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def init_database(database_url: str) -> Engine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    _engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False}
        if database_url.startswith("sqlite")
        else {},
    )
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
