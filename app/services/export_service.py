from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import fields, is_dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Session

from app.models import (
    ExperimentBlock,
    ExperimentEvent,
    ExperimentRound,
    ExperimentSession,
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
    ExperimentSession,
    PersonaProfile,
    PersonaCandidateBatch,
    PersonaInitialization,
    ScreenEvent,
    ExperimentBlock,
    ExperimentRound,
    ExperimentEvent,
    LatentImage,
    Selection,
    FaceRating,
    ProfileChip,
    RecommendationEvaluation,
    FinalSurvey,
    FinalRefinementEvaluation,
]

CANONICAL_EXPORT_ALIASES = {
    ExperimentSession: "sessions.csv",
    ExperimentBlock: "blocks.csv",
    ExperimentRound: "rounds.csv",
    ExperimentEvent: "events.csv",
}

ANALYSIS_ROUND_COLUMNS = [
    "participant_id",
    "session_id",
    "round_status",
    "participant_status",
    "age_band",
    "participant_gender",
    "preferred_target_gender",
    "block_id",
    "block_sequence_index",
    "algorithm",
    "m_value",
    "round_id",
    "shown_image_ids_json",
    "selected_image_id",
    "selected_display_index",
    "preference_rating_1_10",
    "difficulty_rating_1_7",
    "reaction_time_sec",
    "selection_created_at",
    "search_version",
    "proposal_backend",
    "beta",
    "beta_after",
    "effective_sample_size",
    "global_component_mass",
    "min_pairwise_theta_distance",
    "max_pairwise_image_similarity",
    "mean_pairwise_image_similarity",
    "diversity_guard_triggered",
    "num_corrected_queries",
    "max_applied_perturbation",
    "radius_expansion_factor",
    "orthogonal_fallback_used",
    "information_gain_before_correction",
    "information_gain_after_correction",
    "relative_information_gain_loss",
]


def build_csv_export_zip(db: Session) -> bytes:
    archive_buffer = io.BytesIO()
    analysis_rows = build_analysis_round_rows(db)
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
            csv_payload = output.getvalue()
            archive.writestr(f"{model.__tablename__}.csv", csv_payload)
            alias = CANONICAL_EXPORT_ALIASES.get(model)
            if alias:
                archive.writestr(alias, csv_payload)
        archive.writestr(
            "analysis_rounds.csv",
            _rows_to_csv(analysis_rows, ANALYSIS_ROUND_COLUMNS),
        )
        archive.writestr(
            "rounds_flat.csv",
            _rows_to_csv(analysis_rows, ANALYSIS_ROUND_COLUMNS),
        )
        archive.writestr(
            "README.txt",
            (
                "IDEAL-Finder experiment export\n\n"
                "rounds_flat.csv (also available as analysis_rounds.csv for backward "
                "compatibility) contains one analysis-ready row per active-query "
                "selection. The remaining CSV files are normalized raw database tables.\n"
                "JSON-valued columns are preserved as JSON strings. Times are ISO-8601.\n"
                "Binary artifact columns report byte size only; exact bytes remain in DB.\n"
                "Keep this archive encrypted because participant study data may be sensitive.\n"
            ),
        )
    return archive_buffer.getvalue()


def build_analysis_round_rows(db: Session) -> list[dict]:
    rows: list[dict] = []
    selections = list(
        db.scalars(
            select(Selection).order_by(
                Selection.participant_id,
                Selection.created_at,
                Selection.id,
            )
        )
    )
    for selection in selections:
        participant = db.get(Participant, selection.participant_id)
        block = db.get(ExperimentBlock, selection.block_id)
        selected_image = db.get(LatentImage, selection.selected_image_id)
        rating = db.scalar(
            select(FaceRating).where(
                FaceRating.participant_id == selection.participant_id,
                FaceRating.image_id == selection.selected_image_id,
                FaceRating.stage_type == "experiment1_selection",
                FaceRating.batch_id
                == f"{selection.block_id}-round-{selection.round_id:03d}",
            )
        )
        metadata = _json_object(
            selected_image.score_metadata_json if selected_image else "{}"
        )
        query_metrics = metadata.get("query_metrics", {})
        if not isinstance(query_metrics, dict):
            query_metrics = {}
        experiment_round = db.scalar(
            select(ExperimentRound).where(
                ExperimentRound.block_id == selection.block_id,
                ExperimentRound.round_index == selection.round_id,
            )
        )
        rows.append(
            {
                "participant_id": selection.participant_id,
                "session_id": (
                    experiment_round.session_id if experiment_round else ""
                ),
                "round_status": (
                    experiment_round.status if experiment_round else "legacy"
                ),
                "participant_status": participant.status if participant else "",
                "age_band": participant.age_band if participant else "",
                "participant_gender": participant.gender if participant else "",
                "preferred_target_gender": (
                    participant.preferred_target_gender if participant else ""
                ),
                "block_id": selection.block_id,
                "block_sequence_index": block.sequence_index if block else "",
                "algorithm": block.strategy_mode if block else "",
                "m_value": selection.m_value,
                "round_id": selection.round_id,
                "shown_image_ids_json": selection.shown_image_ids,
                "selected_image_id": selection.selected_image_id,
                "selected_display_index": metadata.get("display_index", ""),
                "preference_rating_1_10": (
                    rating.rating_1_10 if rating else ""
                ),
                "difficulty_rating_1_7": selection.difficulty_rating,
                "reaction_time_sec": selection.reaction_time_sec,
                "selection_created_at": selection.created_at,
                "search_version": metadata.get("search_version", ""),
                "proposal_backend": metadata.get("proposal_backend", ""),
                "beta_after": (
                    experiment_round.beta_after if experiment_round else ""
                ),
                **{
                    key: query_metrics.get(key, "")
                    for key in ANALYSIS_ROUND_COLUMNS
                    if key in {
                        "beta",
                        "effective_sample_size",
                        "global_component_mass",
                        "min_pairwise_theta_distance",
                        "max_pairwise_image_similarity",
                        "mean_pairwise_image_similarity",
                        "diversity_guard_triggered",
                        "num_corrected_queries",
                        "max_applied_perturbation",
                        "radius_expansion_factor",
                        "orthogonal_fallback_used",
                        "information_gain_before_correction",
                        "information_gain_after_correction",
                        "relative_information_gain_loss",
                    }
                },
            }
        )
    return rows


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
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bytes):
        return f"<binary:{len(value)} bytes>"
    return value


def _json_object(value: str) -> dict:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rows_to_csv(rows: list[dict], columns: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _serialize(row.get(column)) for column in columns})
    return output.getvalue()
