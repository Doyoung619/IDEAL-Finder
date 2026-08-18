from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_session, init_database
from app.models import ExperimentBlock, LatentImage, Participant
from app.settings import load_config
from app.services.participant_service import experiment_schedule


def main() -> None:
    config = load_config()
    current_version = config.search.version
    init_database(config.database.url)
    invalidated: list[str] = []
    with get_session() as db:
        participants = list(
            db.scalars(
                select(Participant).where(
                    Participant.completed_at.is_(None),
                    Participant.status.not_like("invalidated_%"),
                )
            )
        )
        for participant in participants:
            artifacts = list(
                db.scalars(
                    select(LatentImage).where(
                        LatentImage.participant_id == participant.participant_id
                    )
                )
            )
            blocks = list(
                db.scalars(
                    select(ExperimentBlock).where(
                        ExperimentBlock.participant_id
                        == participant.participant_id
                    )
                )
            )
            has_incompatible_artifact = any(
                json.loads(artifact.score_metadata_json or "{}").get(
                    "search_version"
                )
                != current_version
                for artifact in artifacts
                if artifact.stage_type == "experiment1"
            )
            allowed_algorithms = set(config.experiment.algorithm_order)
            has_incompatible_block = any(
                block.strategy_mode not in allowed_algorithms for block in blocks
            )
            has_incompatible_region_preference = (
                participant.preferred_face_region != "unrestricted"
            )
            has_missing_age_preference = (
                participant.preferred_age_appearance != "twenties_boost"
            )
            try:
                snapshot = yaml.safe_load(
                    Path(participant.config_snapshot_path).read_text(encoding="utf-8")
                )
                has_incompatible_snapshot = (
                    snapshot.get("search", {}).get("version") != current_version
                )
            except (AttributeError, OSError, TypeError, yaml.YAMLError):
                has_incompatible_snapshot = True
            try:
                expected_m_order = [
                    int(item["m"])
                    for item in experiment_schedule(
                        config,
                        participant.participant_id,
                        participant.base_seed,
                    )
                ]
                has_incompatible_schedule = (
                    json.loads(participant.m_condition_order) != expected_m_order
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                has_incompatible_schedule = True
            if (
                has_incompatible_artifact
                or has_incompatible_block
                or has_incompatible_region_preference
                or has_missing_age_preference
                or has_incompatible_schedule
                or has_incompatible_snapshot
            ):
                participant.status = f"invalidated_{current_version}"
                invalidated.append(participant.participant_id)
        db.commit()
    print(
        {
            "search_version": current_version,
            "invalidated_participants": invalidated,
        }
    )


if __name__ == "__main__":
    main()
