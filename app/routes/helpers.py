from __future__ import annotations

from fastapi import Request
from sqlalchemy.orm import Session

from app.models import Participant
from experiments.logging_utils import enter_screen


def current_participant(request: Request, db: Session) -> Participant | None:
    participant_id = request.session.get("participant_id")
    if not participant_id:
        return None
    participant = db.get(Participant, participant_id)
    if participant is not None and participant.status.startswith("invalidated"):
        request.session.clear()
        return None
    return participant


def template_context(
    request: Request,
    participant: Participant | None = None,
    screen_name: str | None = None,
    db: Session | None = None,
    **values,
) -> dict:
    if participant is not None and screen_name and db is not None:
        enter_screen(db, participant.participant_id, screen_name)
    return {
        "request": request,
        "participant": participant,
        "app_name": request.app.state.config.app.name,
        **values,
    }
