from __future__ import annotations

import json
import secrets
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
    # 96 bits of entropy while retaining compatibility with the existing
    # VARCHAR(16) participant key used by deployed databases.
    participant_id = secrets.token_urlsafe(12)
    base_seed = int(config.experiment.seed)
    schedule = experiment_schedule(config, participant_id, base_seed)
    m_values = [item["m"] for item in schedule]
    recommendation_order = (
        []
        if config.persona.enabled
        else assigned_order(
            config.experiment.recommendation_conditions,
            participant_id,
            base_seed + 1,
        )
    )
    run_dir = Path(config.paths.output_dir) / participant_id
    snapshot_path = run_dir / "config.yaml"
    save_config_snapshot(config, snapshot_path)

    participant = Participant(
        participant_id=participant_id,
        consented_at=utc_now(),
        status="consented",
        m_condition_order=json_dumps(m_values),
        recommendation_condition_order=json_dumps(recommendation_order),
        base_seed=base_seed,
        config_snapshot_path=str(snapshot_path),
        consent_version=str(config.app.consent_version),
    )
    db.add(participant)
    db.commit()
    return participant


def m_order(participant: Participant) -> list[int]:
    return [int(value) for value in json.loads(participant.m_condition_order)]


def experiment_schedule(
    config,
    participant_id: str,
    base_seed: int,
) -> list[dict[str, int | str]]:
    """Participant-specific balanced randomization of M and method order."""
    m_values = assigned_order(
        config.experiment.m_list,
        participant_id,
        int(base_seed) + 31,
    )
    schedule: list[dict[str, int | str]] = []
    for index, m_value in enumerate(m_values):
        strategies = assigned_order(
            config.experiment.algorithm_order,
            f"{participant_id}:M={int(m_value)}",
            int(base_seed) + 101 + index,
        )
        schedule.extend(
            {"m": int(m_value), "strategy": str(strategy)}
            for strategy in strategies
        )
    return schedule


def strategy_for_block(config, participant: Participant, sequence_index: int) -> str:
    schedule = experiment_schedule(
        config, participant.participant_id, participant.base_seed
    )
    if sequence_index < 0 or sequence_index >= len(schedule):
        raise IndexError("experiment block index is outside the schedule")
    return str(schedule[sequence_index]["strategy"])


def recommendation_order(participant: Participant) -> list[str]:
    return list(json.loads(participant.recommendation_condition_order))
