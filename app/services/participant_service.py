from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import Participant
from app.settings import save_config_snapshot
from experiments.design import assigned_order
from experiments.logging_utils import json_dumps


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_participant(db: Session, config) -> Participant:
    participant_id = f"ITD-{uuid.uuid4().hex[:8].upper()}"
    base_seed = int(config.experiment.seed)
    m_order = assigned_order(
        config.experiment.m_list,
        participant_id,
        base_seed,
    )
    recommendation_order = assigned_order(
        config.experiment.recommendation_conditions,
        participant_id,
        base_seed + 1,
    )
    run_dir = Path(config.paths.output_dir) / participant_id
    snapshot_path = run_dir / "config.yaml"
    save_config_snapshot(config, snapshot_path)

    participant = Participant(
        participant_id=participant_id,
        consented_at=utc_now(),
        status="consented",
        m_condition_order=json_dumps(m_order),
        recommendation_condition_order=json_dumps(recommendation_order),
        base_seed=base_seed,
        config_snapshot_path=str(snapshot_path),
    )
    db.add(participant)
    db.commit()
    return participant


def m_order(participant: Participant) -> list[int]:
    return [int(value) for value in json.loads(participant.m_condition_order)]


def recommendation_order(participant: Participant) -> list[str]:
    return list(json.loads(participant.recommendation_condition_order))

