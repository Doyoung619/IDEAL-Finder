from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import fields, is_dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.models import (
    ExperimentBlock,
    FaceRating,
    FinalSurvey,
    FinalRefinementEvaluation,
    LatentImage,
    Participant,
    PersonaCandidateBatch,
    PersonaInitialization,
    PersonaProfile,
    ProfileChip,
    RecommendationEvaluation,
    ScreenEvent,
    Selection,
)


EXPORT_MODELS = [
    Participant,
    PersonaProfile,
    PersonaCandidateBatch,
    PersonaInitialization,
    ScreenEvent,
    ExperimentBlock,
    LatentImage,
    Selection,
    FaceRating,
    ProfileChip,
    RecommendationEvaluation,
    FinalSurvey,
    FinalRefinementEvaluation,
]


def build_csv_export_zip(db: Session) -> bytes:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for model in EXPORT_MODELS:
            rows = list(db.scalars(select(model)))
            columns = [column.key for column in inspect(model).columns]
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        column: _serialize(getattr(row, column))
                        for column in columns
                    }
                )
            archive.writestr(f"{model.__tablename__}.csv", output.getvalue())
    return archive_buffer.getvalue()


def write_csv_exports(db: Session, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_csv_export_zip(db))
    return path


def _serialize(value):
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
