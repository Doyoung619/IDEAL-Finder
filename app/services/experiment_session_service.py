from __future__ import annotations

import json
import os
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    ExperimentBlock,
    ExperimentEvent,
    ExperimentRound,
    ExperimentSession,
    LatentImage,
    Participant,
    PersonaInitialization,
    PersonaProfile,
)
from app.services.artifact_storage import ensure_block_files, load_latent
from app.services.participant_service import experiment_schedule, m_order


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def session_for_participant(
    db: Session, participant_id: str
) -> ExperimentSession | None:
    return db.scalar(
        select(ExperimentSession).where(
            ExperimentSession.participant_id == participant_id
        )
    )


def create_experiment_session(
    db: Session,
    config,
    participant: Participant,
    initialization: PersonaInitialization,
    profile: PersonaProfile,
    user_agent: str | None = None,
) -> ExperimentSession:
    existing = session_for_participant(db, participant.participant_id)
    if existing is not None:
        return existing
    theta = np.asarray(np.load(initialization.theta_path), dtype=np.float32)
    schedule = experiment_schedule(
        config, participant.participant_id, participant.base_seed
    )
    session = ExperimentSession(
        session_id=str(uuid.uuid4()),
        participant_id=participant.participant_id,
        persona_completed_at=initialization.confirmed_at or utc_now(),
        status="persona_complete",
        schema_version="experiment_session_v1",
        app_version=os.getenv("IDEAL_APP_VERSION", str(config.search.version)),
        persona_condition={
            "gender": participant.preferred_target_gender,
            "population": str(config.persona.fixed_race),
            "age_range": str(config.persona.fixed_age_range),
            "pool_version": initialization.pool_version,
        },
        persona_data={
            "schema_version": profile.schema_version,
            "responses": _json_value(profile.responses_json, {}),
            "priorities": _json_value(profile.priorities_json, []),
            "prompt_bundle": _json_value(profile.prompt_bundle_json, {}),
            "selected_pool_id": initialization.selected_pool_id,
            "selected_image_id": initialization.selected_image_id,
            "selected_rank": initialization.selected_rank,
            "semantic_score": initialization.semantic_score,
            "initial_rating_1_10": initialization.initial_rating_1_10,
            "selection_confidence_1_7": initialization.selection_confidence_1_7,
        },
        theta_persona=theta.tolist(),
        experiment_seed=participant.base_seed,
        algorithm_order=[str(item["strategy"]) for item in schedule],
        m_order=m_order(participant),
        user_agent=(user_agent or "")[:1024] or None,
    )
    db.add(session)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existing = session_for_participant(db, participant.participant_id)
        if existing is None:
            raise
        return existing
    add_event(
        db,
        session.session_id,
        "session_created",
        payload={"participant_id": participant.participant_id},
    )
    db.commit()
    return session


def mark_session_started(
    db: Session, experiment_session: ExperimentSession, block: ExperimentBlock
) -> None:
    if experiment_session.started_at is None:
        experiment_session.started_at = utc_now()
        experiment_session.status = "in_progress"
        add_event(
            db,
            experiment_session.session_id,
            "session_started",
            block_id=block.block_id,
            payload={"block_index": block.sequence_index},
        )


def mark_session_completed(db: Session, experiment_session: ExperimentSession) -> None:
    if experiment_session.completed_at is None:
        experiment_session.completed_at = utc_now()
        experiment_session.status = "completed"
        add_event(db, experiment_session.session_id, "experiment_completed")


def create_round_submission(
    db: Session,
    runtime,
    experiment_session: ExperimentSession,
    block: ExperimentBlock,
    round_index: int,
    artifacts: list[LatentImage],
    selected_image_id: str,
    preference_rating: int,
    difficulty_rating: int,
    reaction_time_sec: float,
) -> ExperimentRound:
    existing = db.scalar(
        select(ExperimentRound).where(
            ExperimentRound.session_id == experiment_session.session_id,
            ExperimentRound.block_id == block.block_id,
            ExperimentRound.round_index == round_index,
        )
    )
    if existing is not None:
        return existing
    shown_ids = [artifact.image_id for artifact in artifacts]
    selected_index = shown_ids.index(selected_image_id)
    selected = artifacts[selected_index]
    metadata = _json_value(selected.score_metadata_json, {})
    query_metrics = metadata.get("query_metrics", {})
    if not isinstance(query_metrics, dict):
        query_metrics = {}
    final_queries = query_metrics.get("final_theta") or [
        _json_value(artifact.score_metadata_json, {}).get("proposal_feature", [])
        for artifact in artifacts
    ]
    original_queries = query_metrics.get("original_theta") or final_queries
    parameters = _json_value(block.strategy_parameters_json, {})
    prior = runtime.conditional_prior(parameters)
    selected_theta = np.asarray(
        prior.w_to_theta(load_latent(selected)), dtype=np.float32
    )
    expected_information_gain = query_metrics.get(
        "information_gain_after_correction",
        query_metrics.get("mutual_information"),
    )
    query_seed = int(selected.generator_seed) - int(
        metadata.get("display_index", selected_index)
    )
    experiment_round = ExperimentRound(
        round_id=str(uuid.uuid4()),
        session_id=experiment_session.session_id,
        block_id=block.block_id,
        round_index=round_index,
        algorithm=block.strategy_mode,
        m_value=block.m_value,
        status="submitted",
        started_at=min(
            (artifact.created_at for artifact in artifacts), default=utc_now()
        ),
        answered_at=utc_now(),
        response_time_ms=max(0, int(round(reaction_time_sec * 1000))),
        query_points=original_queries,
        final_query_points=final_queries,
        generated_image_ids=shown_ids,
        selected_index=selected_index,
        selected_image_id=selected_image_id,
        selected_theta=selected_theta.tolist(),
        ideal_similarity_rating=preference_rating,
        choice_difficulty_rating=difficulty_rating,
        beta_before=_float_or_none(query_metrics.get("beta")),
        expected_information_gain=_float_or_none(expected_information_gain),
        effective_sample_size=_float_or_none(
            query_metrics.get("effective_sample_size")
        ),
        random_seed=query_seed,
        query_metadata=query_metrics,
    )
    db.add(experiment_round)
    add_event(
        db,
        experiment_session.session_id,
        "choice_submitted",
        block_id=block.block_id,
        payload={
            "round_index": round_index,
            "selected_index": selected_index,
            "selected_image_id": selected_image_id,
        },
    )
    return experiment_round


def complete_round_from_state(
    db: Session, block: ExperimentBlock, experiment_round: ExperimentRound
) -> None:
    ensure_block_files(block)
    with Path(block.strategy_state_path).open("rb") as handle:
        state = pickle.load(handle)
    weights = np.asarray(state.get("weights", []), dtype=np.float64)
    experiment_round.beta_after = _float_or_none(state.get("beta"))
    experiment_round.posterior_mean = np.asarray(
        state.get("map", []), dtype=np.float32
    ).tolist()
    experiment_round.posterior_covariance = np.asarray(
        state.get("covariance", []), dtype=np.float32
    ).tolist()
    if len(weights) and float(np.square(weights).sum()) > 0:
        experiment_round.effective_sample_size = float(
            1.0 / np.square(weights).sum()
        )
    experiment_round.status = "completed"
    experiment_round.completed_at = utc_now()
    add_event(
        db,
        experiment_round.session_id,
        "round_completed",
        block_id=block.block_id,
        payload={"round_index": experiment_round.round_index},
    )
    db.commit()


def mark_round_error(
    db: Session,
    experiment_round: ExperimentRound,
    error: Exception,
) -> None:
    round_id = experiment_round.round_id
    session_id = experiment_round.session_id
    block_id = experiment_round.block_id
    round_index = experiment_round.round_index
    db.rollback()
    persisted_round = db.get(ExperimentRound, round_id)
    if persisted_round is None:
        return
    persisted_round.status = "error"
    add_event(
        db,
        session_id,
        "round_update_failed",
        block_id=block_id,
        payload={
            "round_index": round_index,
            "error_type": type(error).__name__,
        },
    )
    db.commit()


def add_event(
    db: Session,
    session_id: str,
    event_type: str,
    block_id: str | None = None,
    payload: dict | None = None,
) -> ExperimentEvent:
    event = ExperimentEvent(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        block_id=block_id,
        event_type=event_type,
        payload=payload or {},
    )
    db.add(event)
    return event


def _json_value(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _float_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
