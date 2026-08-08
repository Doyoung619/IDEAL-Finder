from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import LatentImage, Participant, Selection
from app.routes.helpers import template_context
from app.services.export_service import build_csv_export_zip


security = HTTPBasic(auto_error=False)


def require_admin(
    credentials: HTTPBasicCredentials | None = Depends(security),
) -> None:
    password = os.getenv("IDEAL_ADMIN_PASSWORD")
    if not password:
        if os.getenv("VERCEL") == "1" or os.getenv("IDEAL_SERVERLESS") == "1":
            raise HTTPException(status_code=404)
        return
    authorized = (
        credentials is not None
        and secrets.compare_digest(credentials.username, "admin")
        and secrets.compare_digest(credentials.password, password)
    )
    if not authorized:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Basic"},
        )


router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@router.get("")
def admin_page(request: Request, db: Session = Depends(get_db)):
    participants = list(
        db.scalars(select(Participant).order_by(Participant.created_at.desc()))
    )
    rows = []
    for participant in participants:
        selection_count = db.scalar(
            select(func.count(Selection.id)).where(
                Selection.participant_id == participant.participant_id
            )
        )
        image_count = db.scalar(
            select(func.count(LatentImage.image_id)).where(
                LatentImage.participant_id == participant.participant_id
            )
        )
        rows.append(
            {
                "participant": participant,
                "selection_count": selection_count or 0,
                "image_count": image_count or 0,
            }
        )
    return request.app.state.templates.TemplateResponse(
        request,
        "admin.html",
        template_context(request, rows=rows),
    )


@router.get("/export.zip")
def export_zip(db: Session = Depends(get_db)):
    payload = build_csv_export_zip(db)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="ideal_type_detector_{timestamp}.zip"'
            )
        },
    )
