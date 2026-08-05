from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_session, init_database
from app.models import ExperimentBlock, LatentImage, Participant
from app.settings import load_config


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
            has_incompatible_block = any(
                block.strategy_mode != config.search.mode for block in blocks
            )
            has_incompatible_region_preference = (
                participant.preferred_face_region != "east_asian_only"
            )
            has_missing_age_preference = not participant.preferred_age_appearance
            if (
                has_incompatible_artifact
                or has_incompatible_block
                or has_incompatible_region_preference
                or has_missing_age_preference
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
