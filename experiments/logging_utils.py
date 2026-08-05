from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ScreenEvent


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads(value: str | None, default=None):
    if not value:
        return default
    return json.loads(value)


def enter_screen(db: Session, participant_id: str, screen_name: str) -> ScreenEvent:
    open_event = db.scalar(
        select(ScreenEvent)
        .where(
            ScreenEvent.participant_id == participant_id,
            ScreenEvent.exit_time.is_(None),
        )
        .order_by(ScreenEvent.enter_time.desc())
    )
    if open_event is not None and open_event.screen_name == screen_name:
        return open_event
    if open_event is not None:
        open_event.exit_time = utc_now()

    event = ScreenEvent(participant_id=participant_id, screen_name=screen_name)
    db.add(event)
    db.commit()
    return event


def exit_screen(db: Session, participant_id: str, screen_name: str) -> None:
    event = db.scalar(
        select(ScreenEvent)
        .where(
            ScreenEvent.participant_id == participant_id,
            ScreenEvent.screen_name == screen_name,
            ScreenEvent.exit_time.is_(None),
        )
        .order_by(ScreenEvent.enter_time.desc())
    )
    if event is not None:
        event.exit_time = utc_now()
        db.commit()

