from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Participant(Base):
    __tablename__ = "participants"

    participant_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    consented_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="created")
    age_band: Mapped[str | None] = mapped_column(String(32))
    gender: Mapped[str | None] = mapped_column(String(32))
    preferred_target_gender: Mapped[str | None] = mapped_column(String(32))
    preferred_face_region: Mapped[str | None] = mapped_column(String(32))
    preferred_age_appearance: Mapped[str | None] = mapped_column(String(32))
    dating_experience: Mapped[str | None] = mapped_column(String(32))
    image_selection_importance: Mapped[int | None] = mapped_column(Integer)
    honest_participation: Mapped[bool] = mapped_column(Boolean, default=False)
    m_condition_order: Mapped[str] = mapped_column(Text)
    recommendation_condition_order: Mapped[str] = mapped_column(Text)
    baseline_latent_path: Mapped[str | None] = mapped_column(Text)
    base_seed: Mapped[int] = mapped_column(Integer)
    config_snapshot_path: Mapped[str | None] = mapped_column(Text)

    blocks: Mapped[list["ExperimentBlock"]] = relationship(
        back_populates="participant", cascade="all, delete-orphan"
    )


class ScreenEvent(Base):
    __tablename__ = "screens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    screen_name: Mapped[str] = mapped_column(String(64), index=True)
    enter_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ExperimentBlock(Base):
    __tablename__ = "experiment_blocks"

    block_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    sequence_index: Mapped[int] = mapped_column(Integer)
    m_value: Mapped[int] = mapped_column(Integer)
    strategy_mode: Mapped[str] = mapped_column(String(32))
    initial_state_id: Mapped[str] = mapped_column(String(64))
    initial_seed: Mapped[int] = mapped_column(Integer)
    mu_path: Mapped[str] = mapped_column(Text)
    sigma: Mapped[float] = mapped_column(Float)
    strategy_state_path: Mapped[str | None] = mapped_column(Text)
    strategy_parameters_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    participant: Mapped[Participant] = relationship(back_populates="blocks")


class LatentImage(Base):
    __tablename__ = "latent_images"

    image_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    block_id: Mapped[str | None] = mapped_column(
        ForeignKey("experiment_blocks.block_id"), index=True
    )
    round_id: Mapped[int | None] = mapped_column(Integer)
    stage_type: Mapped[str] = mapped_column(String(32), index=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    latent_path: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(Text)
    generator_type: Mapped[str] = mapped_column(String(64))
    generator_seed: Mapped[int] = mapped_column(Integer)
    gender_target: Mapped[str | None] = mapped_column(String(32))
    gender_pred: Mapped[str | None] = mapped_column(String(32))
    gender_confidence: Mapped[float | None] = mapped_column(Float)
    face_quality_score: Mapped[float | None] = mapped_column(Float)
    condition_type: Mapped[str | None] = mapped_column(String(64), index=True)
    score_metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Selection(Base):
    __tablename__ = "selections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    block_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_blocks.block_id"), index=True
    )
    round_id: Mapped[int] = mapped_column(Integer)
    m_value: Mapped[int] = mapped_column(Integer)
    shown_image_ids: Mapped[str] = mapped_column(Text)
    selected_image_id: Mapped[str] = mapped_column(
        ForeignKey("latent_images.image_id")
    )
    reaction_time_sec: Mapped[float] = mapped_column(Float)
    difficulty_rating: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FaceRating(Base):
    __tablename__ = "face_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    stage_type: Mapped[str] = mapped_column(String(32), index=True)
    image_id: Mapped[str] = mapped_column(
        ForeignKey("latent_images.image_id"), index=True
    )
    rating_1_10: Mapped[int] = mapped_column(Integer)
    batch_id: Mapped[str | None] = mapped_column(String(64))
    condition_type: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ProfileChip(Base):
    __tablename__ = "profile_chips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    category: Mapped[str] = mapped_column(String(64))
    selected_option: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RecommendationEvaluation(Base):
    __tablename__ = "recommendation_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    batch_id: Mapped[str] = mapped_column(String(64), unique=True)
    condition_type: Mapped[str] = mapped_column(String(64), index=True)
    shown_image_ids: Mapped[str] = mapped_column(Text)
    contains_ideal_type: Mapped[bool] = mapped_column(Boolean)
    selected_image_id: Mapped[str] = mapped_column(
        ForeignKey("latent_images.image_id")
    )
    selected_rating_1_10: Mapped[int] = mapped_column(Integer)
    reaction_time_sec: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FinalSurvey(Base):
    __tablename__ = "final_survey"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), unique=True, index=True
    )
    q1: Mapped[int] = mapped_column(Integer)
    q2: Mapped[int] = mapped_column(Integer)
    q3: Mapped[int] = mapped_column(Integer)
    q4: Mapped[int] = mapped_column(Integer)
    free_text: Mapped[str | None] = mapped_column(Text)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class PersonaProfile(Base):
    __tablename__ = "persona_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), unique=True, index=True
    )
    schema_version: Mapped[str] = mapped_column(String(32))
    target_gender: Mapped[str] = mapped_column(String(16))
    fixed_race: Mapped[str] = mapped_column(String(32))
    fixed_age_range: Mapped[str] = mapped_column(String(32))
    responses_json: Mapped[str] = mapped_column(Text)
    priorities_json: Mapped[str] = mapped_column(Text)
    korean_summary: Mapped[str] = mapped_column(Text)
    full_prompt_en: Mapped[str] = mapped_column(Text)
    prompt_bundle_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class PersonaCandidateBatch(Base):
    __tablename__ = "persona_candidate_batches"

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), index=True
    )
    page_index: Mapped[int] = mapped_column(Integer)
    pool_version: Mapped[str] = mapped_column(String(128))
    shown_pool_ids_json: Mapped[str] = mapped_column(Text)
    shown_image_ids_json: Mapped[str] = mapped_column(Text)
    semantic_scores_json: Mapped[str] = mapped_column(Text)
    semantic_ranks_json: Mapped[str] = mapped_column(Text)
    mmr_scores_json: Mapped[str] = mapped_column(Text)
    display_order_json: Mapped[str] = mapped_column(Text)
    action: Mapped[str | None] = mapped_column(String(32))
    selected_pool_id: Mapped[str | None] = mapped_column(String(128))
    selected_image_id: Mapped[str | None] = mapped_column(
        ForeignKey("latent_images.image_id")
    )
    reaction_time_sec: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PersonaInitialization(Base):
    __tablename__ = "persona_initializations"

    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), primary_key=True
    )
    persona_profile_id: Mapped[int] = mapped_column(ForeignKey("persona_profiles.id"))
    candidate_batch_id: Mapped[str] = mapped_column(
        ForeignKey("persona_candidate_batches.batch_id")
    )
    selected_pool_id: Mapped[str] = mapped_column(String(128))
    selected_image_id: Mapped[str] = mapped_column(ForeignKey("latent_images.image_id"))
    selected_rank: Mapped[int] = mapped_column(Integer)
    semantic_score: Mapped[float] = mapped_column(Float)
    theta_path: Mapped[str] = mapped_column(Text)
    w_path: Mapped[str] = mapped_column(Text)
    initial_rating_1_10: Mapped[int] = mapped_column(Integer)
    selection_confidence_1_7: Mapped[int] = mapped_column(Integer)
    pool_version: Mapped[str] = mapped_column(String(128))
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class FinalRefinementEvaluation(Base):
    __tablename__ = "final_refinement_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.participant_id"), unique=True, index=True
    )
    initial_image_id: Mapped[str] = mapped_column(ForeignKey("latent_images.image_id"))
    final_map_image_id: Mapped[str] = mapped_column(ForeignKey("latent_images.image_id"))
    last_winner_image_id: Mapped[str | None] = mapped_column(ForeignKey("latent_images.image_id"))
    display_order_json: Mapped[str] = mapped_column(Text)
    preferred_image_id: Mapped[str] = mapped_column(ForeignKey("latent_images.image_id"))
    initial_rating_1_10: Mapped[int] = mapped_column(Integer)
    final_rating_1_10: Mapped[int] = mapped_column(Integer)
    perceived_improvement_1_7: Mapped[int] = mapped_column(Integer)
    final_match_1_10: Mapped[int] = mapped_column(Integer)
    reaction_time_sec: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
