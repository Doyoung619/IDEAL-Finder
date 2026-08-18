from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import (
    FinalRefinementEvaluation,
    LatentImage,
    Participant,
    PersonaCandidateBatch,
    PersonaInitialization,
    PersonaProfile,
    Selection,
)
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
                "persona_complete": db.get(PersonaInitialization, participant.participant_id) is not None,
                "persona_pages": db.scalar(
                    select(func.count(PersonaCandidateBatch.batch_id)).where(
                        PersonaCandidateBatch.participant_id == participant.participant_id
                    )
                ) or 0,
                "final_complete": db.scalar(
                    select(func.count(FinalRefinementEvaluation.id)).where(
                        FinalRefinementEvaluation.participant_id == participant.participant_id
                    )
                ) or 0,
            }
        )
    readiness = []
    minimum = int(request.app.state.config.persona_pool.minimum_usable_size)
    for gender, directory in (
        ("female", request.app.state.config.persona.female_pool),
        ("male", request.app.state.config.persona.male_pool),
    ):
        root = Path(directory)
        metadata_path = root / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        size = int(metadata.get("size", 0))
        readiness.append(
            {
                "gender": gender,
                "path": str(root),
                "size": size,
                "ready": (root / "pool.npz").exists() and size >= minimum,
                "version": metadata.get("pool_version", "—"),
            }
        )
    return request.app.state.templates.TemplateResponse(
        request,
        "admin.html",
        template_context(
            request,
            rows=rows,
            readiness=readiness,
            minimum_pool_size=minimum,
            require_real_clip=request.app.state.config.persona.require_real_clip,
        ),
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
