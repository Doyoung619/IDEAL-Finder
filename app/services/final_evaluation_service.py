from __future__ import annotations

import json
import pickle
import shutil
import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ExperimentBlock,
    FinalRefinementEvaluation,
    LatentImage,
    Participant,
    PersonaInitialization,
    Selection,
)
from experiments.design import stable_seed
from experiments.logging_utils import json_loads


def prepare_final_artifacts(
    db: Session, runtime, participant: Participant
) -> tuple[LatentImage, LatentImage, LatentImage | None, list[LatentImage]]:
    initialization = db.get(PersonaInitialization, participant.participant_id)
    if initialization is None:
        raise RuntimeError("Persona initialization is required for final evaluation")
    initial = db.get(LatentImage, initialization.selected_image_id)
    if initial is None:
        raise RuntimeError("Initial persona image artifact is missing")
    blocks = list(
        db.scalars(
            select(ExperimentBlock)
            .where(ExperimentBlock.participant_id == participant.participant_id)
            .order_by(ExperimentBlock.sequence_index)
        )
    )
    if not blocks or any(block.completed_at is None for block in blocks):
        raise RuntimeError("Interactive refinement is not complete")
    final_block = blocks[-1]
    final_map = db.scalar(
        select(LatentImage).where(
            LatentImage.participant_id == participant.participant_id,
            LatentImage.stage_type == "final_map",
        )
    )
    if final_map is None:
        with Path(final_block.strategy_state_path).open("rb") as handle:
            state = pickle.load(handle)
        final_theta = np.asarray(state["map"], dtype=np.float32)
        parameters = json_loads(final_block.strategy_parameters_json, {})
        prior = runtime.conditional_prior(parameters)
        final_w = np.asarray(prior.theta_to_w(final_theta), dtype=np.float32)
        final_image = runtime.generator.decode(np.atleast_2d(final_w))[0]
        cache_dir = (
            Path(runtime.config.paths.cache_dir)
            / participant.participant_id
            / "final"
        )
        output_dir = (
            Path(runtime.config.paths.output_dir)
            / participant.participant_id
            / "final"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_theta = cache_dir / "final_theta.npy"
        cache_w = cache_dir / "final_w.npy"
        cache_image = cache_dir / "final_face.png"
        np.save(cache_theta, final_theta)
        np.save(cache_w, final_w)
        final_image.save(cache_image, format="PNG", optimize=True)
        np.save(output_dir / "final_theta.npy", final_theta)
        np.save(output_dir / "final_w.npy", final_w)
        shutil.copy2(cache_image, output_dir / "final_face.png")
        shutil.copy2(initial.image_path, output_dir / "initial_face.png")
        final_map = LatentImage(
            image_id=str(uuid.uuid4()),
            participant_id=participant.participant_id,
            block_id=final_block.block_id,
            round_id=None,
            stage_type="final_map",
            batch_id=f"{participant.participant_id}-final",
            latent_path=str(cache_w),
            image_path=str(cache_image),
            generator_type=runtime.generator.generator_name,
            generator_seed=final_block.initial_seed,
            gender_target=participant.preferred_target_gender,
            gender_pred=participant.preferred_target_gender,
            gender_confidence=None,
            face_quality_score=None,
            condition_type="posterior_map",
            score_metadata_json=json.dumps(
                {
                    "source": "posterior_map",
                    "theta_path": str(cache_theta),
                    "state_path": final_block.strategy_state_path,
                    "warm_start_pool_id": initialization.selected_pool_id,
                },
                sort_keys=True,
            ),
        )
        db.add(final_map)
        (output_dir / "final_metadata.json").write_text(
            json.dumps(
                {
                    "final_image_id": final_map.image_id,
                    "initial_image_id": initial.image_id,
                    "final_block_id": final_block.block_id,
                    "state_path": final_block.strategy_state_path,
                    "persona_pool_id": initialization.selected_pool_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        db.flush()

    last_selection = db.scalar(
        select(Selection)
        .where(Selection.participant_id == participant.participant_id)
        .order_by(Selection.created_at.desc(), Selection.id.desc())
    )
    last_winner = (
        db.get(LatentImage, last_selection.selected_image_id)
        if last_selection is not None
        else None
    )
    output_dir = Path(runtime.config.paths.output_dir) / participant.participant_id / "final"
    if last_winner is not None:
        shutil.copy2(last_winner.image_path, output_dir / "last_winner_face.png")

    unique: list[LatentImage] = []
    unique_latents: list[np.ndarray] = []
    for artifact in (initial, final_map, last_winner):
        if artifact is None:
            continue
        latent = np.asarray(np.load(artifact.latent_path), dtype=np.float32)
        if any(
            latent.shape == previous.shape and np.allclose(latent, previous, atol=1e-6)
            for previous in unique_latents
        ):
            continue
        unique.append(artifact)
        unique_latents.append(latent)
    seed = stable_seed(f"{participant.participant_id}:final-display", participant.base_seed)
    random = np.random.default_rng(seed)
    display = [unique[index] for index in random.permutation(len(unique))]
    db.commit()
    return initial, final_map, last_winner, display


def save_final_evaluation(
    db: Session,
    participant: Participant,
    initial: LatentImage,
    final_map: LatentImage,
    last_winner: LatentImage | None,
    display: list[LatentImage],
    preferred_image_id: str,
    initial_rating: int,
    final_rating: int,
    perceived_improvement: int,
    final_match: int,
    reaction_time_sec: float,
) -> FinalRefinementEvaluation:
    existing = db.scalar(
        select(FinalRefinementEvaluation).where(
            FinalRefinementEvaluation.participant_id == participant.participant_id
        )
    )
    if existing is not None:
        return existing
    shown_ids = [artifact.image_id for artifact in display]
    if preferred_image_id not in shown_ids:
        raise ValueError("Preferred image must be one of the displayed final cards")
    if not (
        1 <= int(initial_rating) <= 10
        and 1 <= int(final_rating) <= 10
        and 1 <= int(perceived_improvement) <= 7
        and 1 <= int(final_match) <= 10
    ):
        raise ValueError("Final evaluation ratings are outside their allowed ranges")
    evaluation = FinalRefinementEvaluation(
        participant_id=participant.participant_id,
        initial_image_id=initial.image_id,
        final_map_image_id=final_map.image_id,
        last_winner_image_id=last_winner.image_id if last_winner else None,
        display_order_json=json.dumps(shown_ids),
        preferred_image_id=preferred_image_id,
        initial_rating_1_10=int(initial_rating),
        final_rating_1_10=int(final_rating),
        perceived_improvement_1_7=int(perceived_improvement),
        final_match_1_10=int(final_match),
        reaction_time_sec=max(0.0, float(reaction_time_sec)),
    )
    db.add(evaluation)
    participant.status = "final_evaluation_complete"
    db.commit()
    return evaluation
