from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import LatentImage, Participant, Selection
from app.routes.helpers import template_context
from app.services.export_service import build_csv_export_zip


router = APIRouter(prefix="/admin")


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

