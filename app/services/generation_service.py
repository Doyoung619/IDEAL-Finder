from __future__ import annotations

import json
import logging
import pickle
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ExperimentBlock,
    LatentImage,
    Participant,
    PersonaInitialization,
    ProfileChip,
    Selection,
)
from app.services.artifact_storage import (
    array_to_npy_bytes,
    ensure_block_files,
    image_to_png_bytes,
    load_latent,
    persist_artifacts_in_db,
    sync_block_files,
)
from app.services.participant_service import m_order
from app.services.experiment_session_service import (
    add_event,
    mark_session_started,
    session_for_participant,
)
from app.services.profile_service import build_profile_prompt
from core.latent_sampler import farthest_point_sampling
from core.preference_model import PairwisePreferenceModel
from core.query_diversity import QueryDiversityConfig, ensure_query_diversity
from core.utils import latent_fingerprint
from experiments.design import stable_seed
from experiments.logging_utils import json_dumps, json_loads


logger = logging.getLogger("ideal_finder.generation")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class EvaluatedCandidate:
    latent: np.ndarray
    image: Image.Image
    gender_label: str
    gender_confidence: float
    gender_backend: str
    is_adult: bool
    adult_confidence: float
    adult_backend: str
    is_clean_portrait: bool
    portrait_confidence: float
    portrait_backend: str
    face_region_label: str
    east_asian_confidence: float
    face_region_backend: str
    age_appearance_label: str
    twenties_confidence: float
    twenties_thirties_confidence: float
    age_appearance_backend: str
    quality_score: float
    quality_accepted: bool
    face_detected: bool
    quality_backend: str


def ensure_participant_baseline(
    db: Session,
    runtime,
    participant: Participant,
) -> np.ndarray:
    if participant.baseline_latent_path:
        return np.load(participant.baseline_latent_path)

    seed = stable_seed(participant.participant_id, participant.base_seed)
    accepted = generate_filtered_prior_candidates(
        runtime,
        participant.preferred_target_gender,
        "unrestricted",
        participant.preferred_age_appearance,
        count=8,
        seed=seed,
    )
    chosen = max(
        accepted,
        key=lambda item: _demographic_preference_score(
            item,
            "unrestricted",
            participant.preferred_age_appearance,
        ),
    )
    participant_dir = Path(runtime.config.paths.output_dir) / participant.participant_id
    latent_path = participant_dir / "baseline_w.npy"
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(latent_path, chosen.latent.astype(np.float32))
    metadata_path = participant_dir / "baseline_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "initial_state_id": latent_fingerprint(chosen.latent),
                "target_gender": participant.preferred_target_gender,
                "gender_pred": chosen.gender_label,
                "gender_confidence": chosen.gender_confidence,
                "gender_backend": chosen.gender_backend,
                "adult_confidence": chosen.adult_confidence,
                "adult_backend": chosen.adult_backend,
                "portrait_confidence": chosen.portrait_confidence,
                "portrait_backend": chosen.portrait_backend,
                "face_region_label": chosen.face_region_label,
                "east_asian_confidence": chosen.east_asian_confidence,
                "face_region_backend": chosen.face_region_backend,
                "age_appearance_label": chosen.age_appearance_label,
                "twenties_confidence": chosen.twenties_confidence,
                "twenties_thirties_confidence": (
                    chosen.twenties_thirties_confidence
                ),
                "age_appearance_backend": chosen.age_appearance_backend,
                "quality_score": chosen.quality_score,
                "face_detected": chosen.face_detected,
                "quality_backend": chosen.quality_backend,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    participant.baseline_latent_path = str(latent_path)
    db.commit()
    return chosen.latent


def get_or_create_block(
    db: Session,
    runtime,
    participant: Participant,
    sequence_index: int,
    strategy_mode: str | None = None,
    strategy_parameters: dict | None = None,
) -> ExperimentBlock:
    values = m_order(participant)
    m_value = values[sequence_index]
    block_id = f"{participant.participant_id}-B{sequence_index + 1}-M{m_value}"
    existing = db.get(ExperimentBlock, block_id)
    if existing is not None:
        return existing
    selected_mode = strategy_mode or "entropy"
    if selected_mode not in set(runtime.config.experiment.algorithm_order):
        raise ValueError(f"Unsupported query algorithm: {selected_mode}")
    strategy_parameters = strategy_parameters or {}

    conditional_prior = runtime.conditional_prior(strategy_parameters)
    initialization = db.get(PersonaInitialization, participant.participant_id)
    if initialization is None and runtime.config.persona.required:
        raise RuntimeError(
            "Persona warm start is required before the experiment can begin."
        )
    experiment_session = session_for_participant(db, participant.participant_id)
    if initialization is not None:
        if Path(initialization.theta_path).exists():
            initial_theta = np.asarray(
                np.load(initialization.theta_path), dtype=np.float32
            )
        elif experiment_session is not None:
            initial_theta = np.asarray(
                experiment_session.theta_persona, dtype=np.float32
            )
        else:
            raise FileNotFoundError("Persona theta artifact is unavailable")
        if Path(initialization.w_path).exists():
            baseline = np.asarray(np.load(initialization.w_path), dtype=np.float32)
        else:
            selected = db.get(LatentImage, initialization.selected_image_id)
            if selected is None:
                raise FileNotFoundError("Persona W artifact is unavailable")
            baseline = np.asarray(load_latent(selected), dtype=np.float32)
    else:
        initial_theta = conditional_prior.theta_mean.astype(np.float32)
        baseline = np.asarray(
            conditional_prior.theta_to_w(initial_theta), dtype=np.float32
        )
    if initial_theta.shape != (conditional_prior.dimension,):
        raise ValueError("Persona theta dimension does not match the conditional prior")
    expected_w = np.asarray(conditional_prior.theta_to_w(initial_theta), dtype=np.float32)
    if baseline.shape != expected_w.shape or not np.allclose(baseline, expected_w, atol=1e-5):
        raise ValueError("Persona W is not the exact transform of the selected theta")
    block_dir = (
        Path(runtime.config.paths.output_dir)
        / participant.participant_id
        / "blocks"
        / f"block_{sequence_index + 1}_m{m_value}"
    )
    mu_path = block_dir / "current_mu.npy"
    state_path = block_dir / "strategy_state.pkl"
    block_dir.mkdir(parents=True, exist_ok=True)
    np.save(mu_path, baseline.astype(np.float32))
    np.save(block_dir / "initial_theta.npy", initial_theta.astype(np.float32))
    np.save(block_dir / "initial_w.npy", baseline.astype(np.float32))
    covariance_scale = float(runtime.config.persona.prior_covariance_scale)
    strategy_parameters = {
        **strategy_parameters,
        "warm_start_required": bool(runtime.config.persona.required),
        "warm_start_type": "persona" if initialization is not None else "legacy_prior",
        "warm_start_pool_id": (
            initialization.selected_pool_id if initialization is not None else None
        ),
        "warm_start_profile_version": (
            runtime.config.persona.schema_version if initialization is not None else None
        ),
        "prior_covariance_scale": covariance_scale,
    }
    strategy = runtime.strategy_for(selected_mode, strategy_parameters)
    strategy.initialize_state(
        str(state_path),
        initial_mean=initial_theta,
        covariance_scale=covariance_scale,
        metadata={
            "participant_id": participant.participant_id,
            "selected_pool_id": (
                initialization.selected_pool_id if initialization is not None else None
            ),
            "persona_schema_version": (
                runtime.config.persona.schema_version if initialization is not None else None
            ),
        },
    )
    initial_seed = stable_seed(
        f"{participant.participant_id}:shared-mixture-initial",
        participant.base_seed,
    )
    block = ExperimentBlock(
        block_id=block_id,
        participant_id=participant.participant_id,
        session_id=(experiment_session.session_id if experiment_session else None),
        sequence_index=sequence_index,
        m_value=m_value,
        strategy_mode=selected_mode,
        initial_state_id=latent_fingerprint(baseline),
        initial_seed=initial_seed,
        mu_path=str(mu_path),
        mu_data=array_to_npy_bytes(baseline),
        sigma=1.0,
        strategy_state_path=str(state_path),
        strategy_parameters_json=json_dumps(strategy_parameters),
    )
    db.add(block)
    db.flush()
    if experiment_session is not None:
        mark_session_started(db, experiment_session, block)
        add_event(
            db,
            experiment_session.session_id,
            "block_started",
            block_id=block.block_id,
            payload={
                "block_index": sequence_index,
                "algorithm": selected_mode,
                "m_value": m_value,
            },
        )
    if initialization is not None:
        participant.baseline_latent_path = initialization.w_path
    db.commit()
    return block


def find_block(
    db: Session,
    participant: Participant,
    sequence_index: int,
) -> ExperimentBlock | None:
    values = m_order(participant)
    if sequence_index < 0 or sequence_index >= len(values):
        return None
    block_id = (
        f"{participant.participant_id}-B{sequence_index + 1}-M{values[sequence_index]}"
    )
    return db.get(ExperimentBlock, block_id)


def generate_experiment_round(
    db: Session,
    runtime,
    participant: Participant,
    block: ExperimentBlock,
    round_id: int,
) -> list[LatentImage]:
    existing = list(
        db.scalars(
            select(LatentImage)
            .where(
                LatentImage.participant_id == participant.participant_id,
                LatentImage.block_id == block.block_id,
                LatentImage.round_id == round_id,
                LatentImage.stage_type == "experiment1",
            )
            .order_by(LatentImage.created_at)
        )
    )
    if existing:
        return existing
    with runtime.generation_lock:
        existing = list(
            db.scalars(
                select(LatentImage)
                .where(
                    LatentImage.participant_id == participant.participant_id,
                    LatentImage.block_id == block.block_id,
                    LatentImage.round_id == round_id,
                    LatentImage.stage_type == "experiment1",
                )
                .order_by(LatentImage.created_at)
            )
        )
        if existing:
            return existing
        return _generate_query_round(
            db=db,
            runtime=runtime,
            participant=participant,
            block=block,
            round_id=round_id,
        )


def _generate_query_round(
    db: Session,
    runtime,
    participant: Participant,
    block: ExperimentBlock,
    round_id: int,
) -> list[LatentImage]:
    """Optimize, decode, persist, and return one direct synthetic query set."""
    total_started = time.perf_counter()
    ensure_block_files(block)
    strategy_parameters = json_loads(block.strategy_parameters_json or "{}")
    strategy = runtime.strategy_for(block.strategy_mode, strategy_parameters)
    center = np.load(block.mu_path)
    seed = (
        block.initial_seed
        if round_id == 1
        else stable_seed(
            f"{block.block_id}:round:{round_id}", participant.base_seed
        )
    )
    algorithm_started = time.perf_counter()
    proposal = strategy.propose(
        center=center,
        sigma=1.0,
        count=block.m_value,
        seed=seed,
        state_path=block.strategy_state_path,
        round_id=round_id,
        display_count=block.m_value,
    )
    if len(proposal.latents) != block.m_value:
        raise RuntimeError(
            f"{block.strategy_mode} returned an unexpected number of synthetic queries"
        )
    round_directory = (
        Path(block.strategy_state_path).parent / f"round_{round_id:02d}"
    )
    covariance_path = round_directory / "posterior_covariance.npy"
    if not covariance_path.exists():
        raise RuntimeError("Query strategy did not persist posterior covariance")
    conditional_prior = runtime.conditional_prior(strategy_parameters)
    with Path(block.strategy_state_path).open("rb") as state_handle:
        posterior_state = pickle.load(state_handle)
    final_theta, diversity_metrics = ensure_query_diversity(
        theta=np.asarray(proposal.features, dtype=np.float32),
        posterior_covariance=np.load(covariance_path),
        prior=conditional_prior,
        generator=runtime.generator,
        image_embedder=runtime.clip_ranker,
        config=QueryDiversityConfig(
            **runtime.config.query_diversity.as_dict()
        ),
        posterior_particles=np.asarray(posterior_state["particles"]),
        posterior_weights=np.asarray(posterior_state["weights"]),
        beta=float(posterior_state.get("beta", 1.0)),
        algorithm=block.strategy_mode,
    )
    proposal.features = final_theta
    proposal.latents = np.asarray(
        conditional_prior.theta_to_w(final_theta), dtype=np.float32
    )
    proposal.metadata = {**(proposal.metadata or {}), **diversity_metrics}
    algorithm_latency_ms = (time.perf_counter() - algorithm_started) * 1000.0
    sync_block_files(block)
    generation_started = time.perf_counter()
    evaluated = evaluate_candidates(runtime, proposal.latents)
    generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0
    np.save(round_directory / "final_query_points.npy", final_theta)
    image_directory = round_directory / "decoded_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[LatentImage] = []
    for display_index, candidate in enumerate(evaluated):
        candidate.image.save(
            image_directory / f"query_{display_index + 1:02d}.png",
            format="PNG",
            optimize=True,
        )
        metadata = {
            "search_version": runtime.config.search.version,
            "query_algorithm": block.strategy_mode,
            "proposal_backend": proposal.backend,
            "proposal_feature": proposal.features[display_index].tolist(),
            "display_index": display_index,
            "proposal_role": (
                proposal.roles[display_index] if proposal.roles else None
            ),
            "query_metrics": proposal.metadata,
        }
        artifacts.append(
            persist_candidate(
                db=db,
                runtime=runtime,
                participant=participant,
                candidate=candidate,
                stage_type="experiment1",
                condition_type=f"{block.strategy_mode}|M={block.m_value}",
                seed=seed + display_index,
                block_id=block.block_id,
                round_id=round_id,
                batch_id=None,
                metadata=metadata,
            )
        )
    experiment_session = session_for_participant(db, participant.participant_id)
    if experiment_session is not None:
        add_event(
            db,
            experiment_session.session_id,
            "query_generated",
            block_id=block.block_id,
            payload={
                "round_index": round_id,
                "algorithm": block.strategy_mode,
                "m_value": block.m_value,
                "image_ids": [artifact.image_id for artifact in artifacts],
            },
        )
    db.commit()
    total_request_latency_ms = (time.perf_counter() - total_started) * 1000.0
    logger.info(
        "query_latency algorithm=%s m_value=%s round=%s "
        "algorithm_latency_ms=%.1f generation_latency_ms=%.1f "
        "total_request_latency_ms=%.1f",
        block.strategy_mode,
        block.m_value,
        round_id,
        algorithm_latency_ms,
        generation_latency_ms,
        total_request_latency_ms,
    )
    return artifacts


def update_block_after_selection(
    db: Session,
    runtime,
    block: ExperimentBlock,
    shown_image_ids: list[str],
    selected_image_id: str,
    round_id: int,
    preference_rating: int | None = None,
    difficulty_rating: int | None = None,
) -> None:
    ensure_block_files(block)
    strategy_parameters = json_loads(block.strategy_parameters_json or "{}")
    strategy = runtime.strategy_for(block.strategy_mode, strategy_parameters)
    shown = [db.get(LatentImage, image_id) for image_id in shown_image_ids]
    selected = db.get(LatentImage, selected_image_id)
    if selected is None or any(image is None for image in shown):
        raise ValueError("Selection refers to unknown latent images")
    shown_latents = np.vstack([load_latent(image) for image in shown])
    winner_latent = load_latent(selected)
    center = np.load(block.mu_path)
    new_center, new_sigma = strategy.update(
        center=center,
        sigma=block.sigma,
        shown_latents=shown_latents,
        winner_latent=winner_latent,
        state_path=block.strategy_state_path,
        preference_rating=preference_rating,
        difficulty_rating=difficulty_rating,
    )
    np.save(block.mu_path, new_center.astype(np.float32))
    block.sigma = new_sigma
    sync_block_files(block)
    db.commit()


def get_or_generate_answer_key_image(
    db: Session,
    runtime,
    participant: Participant,
    index: int,
) -> LatentImage:
    batch_id = f"{participant.participant_id}-answer-{index:03d}"
    existing = db.scalar(
        select(LatentImage).where(LatentImage.batch_id == batch_id)
    )
    if existing is not None:
        return existing
    seed = stable_seed(batch_id, participant.base_seed)
    candidate = generate_filtered_prior_candidates(
        runtime,
        participant.preferred_target_gender,
        participant.preferred_face_region,
        participant.preferred_age_appearance,
        count=1,
        seed=seed,
    )[0]
    artifact = persist_candidate(
        db=db,
        runtime=runtime,
        participant=participant,
        candidate=candidate,
        stage_type="answer_key",
        condition_type="answer_key",
        seed=seed,
        block_id=None,
        round_id=index,
        batch_id=batch_id,
        metadata={},
    )
    db.commit()
    return artifact


def get_or_generate_recommendation_set(
    db: Session,
    runtime,
    participant: Participant,
    condition_type: str,
    batch_index: int,
) -> tuple[str, list[LatentImage], str]:
    batch_id = (
        f"{participant.participant_id}-recommendation-"
        f"{condition_type}-{batch_index:02d}"
    )
    existing = list(
        db.scalars(
            select(LatentImage)
            .where(LatentImage.batch_id == batch_id)
            .order_by(LatentImage.created_at)
        )
    )
    if existing:
        metadata = json_loads(existing[0].score_metadata_json, {})
        return batch_id, existing, metadata.get("prompt", "")

    runtime.ensure_ready()
    set_size = runtime.config.experiment.recommendation_set_size
    pool_size = max(
        runtime.config.experiment.candidate_pool_size,
        set_size * runtime.config.experiment.candidate_pool_multiplier,
    )
    pool_size = _region_pool_size(
        runtime,
        participant.preferred_face_region,
        participant.preferred_age_appearance,
        pool_size,
        required_count=set_size,
    )
    seed = stable_seed(batch_id, participant.base_seed)
    latents = runtime.generator.sample_prior(pool_size, seed)
    features = runtime.projector.transform(latents)
    evaluated = evaluate_candidates(runtime, latents)
    candidate_indices, relaxed = accepted_candidate_indices(
        runtime,
        evaluated,
        participant.preferred_target_gender,
        required=set_size,
    )

    prompt = ""
    if condition_type == "none":
        random = np.random.default_rng(seed)
        scores = np.asarray(
            [
                evaluated[index].quality_score + random.normal(0, 0.01)
                for index in candidate_indices
            ],
            dtype=np.float32,
        )
        rank_backend = "gender_prior_quality"
    elif condition_type == "basic_info":
        target = {
            "male": "adult man",
            "female": "adult woman",
        }.get(participant.preferred_target_gender, "adult person")
        prompt = (
            f"high quality realistic studio portrait photograph of an {target}, "
            "front-facing, natural expression"
        )
        subset_images = [evaluated[index].image for index in candidate_indices]
        scores = runtime.clip_ranker.score(
            subset_images,
            prompt,
            features[candidate_indices],
        )
        rank_backend = runtime.clip_ranker.backend
    elif condition_type == "click_profile":
        chips = list(
            db.scalars(
                select(ProfileChip).where(
                    ProfileChip.participant_id == participant.participant_id
                )
            )
        )
        prompt = build_profile_prompt(
            participant.preferred_target_gender,
            [chip.selected_option for chip in chips],
        )
        subset_images = [evaluated[index].image for index in candidate_indices]
        scores = runtime.clip_ranker.score(
            subset_images,
            prompt,
            features[candidate_indices],
        )
        rank_backend = runtime.clip_ranker.backend
    elif condition_type == "selection_history":
        preference_model = fit_participant_preference_model(
            db,
            runtime,
            participant,
        )
        if preference_model is None:
            scores = np.asarray(
                [evaluated[index].quality_score for index in candidate_indices],
                dtype=np.float32,
            )
            rank_backend = "quality_fallback_no_pairwise_data"
        else:
            scores = preference_model.score(latents[candidate_indices])
            rank_backend = "pairwise_logistic_regression"
    else:
        raise ValueError(f"Unknown recommendation condition: {condition_type}")

    order = np.argsort(scores)[::-1]
    shortlist_size = min(len(order), max(set_size * 2, set_size))
    shortlist_local = order[:shortlist_size]
    shortlist_global = [candidate_indices[index] for index in shortlist_local]
    diversity_local = farthest_point_sampling(
        features[shortlist_global],
        count=set_size,
        seed=seed + 11,
        priority_scores=scores[shortlist_local],
    )
    selected_indices = [shortlist_global[index] for index in diversity_local]
    ranked_indices = [candidate_indices[index] for index in order]
    selected_indices = region_boosted_candidate_indices(
        runtime,
        evaluated,
        ranked_indices,
        required=set_size,
        preference=participant.preferred_face_region,
        age_preference=participant.preferred_age_appearance,
        preserve_indices=[],
        fallback_order=selected_indices,
    )
    score_by_global = {
        candidate_indices[local_index]: float(score)
        for local_index, score in enumerate(scores)
    }

    artifacts: list[LatentImage] = []
    for display_index, candidate_index in enumerate(selected_indices):
        artifact = persist_candidate(
            db=db,
            runtime=runtime,
            participant=participant,
            candidate=evaluated[candidate_index],
            stage_type="recommendation",
            condition_type=condition_type,
            seed=seed + candidate_index,
            block_id=None,
            round_id=batch_index,
            batch_id=batch_id,
            metadata={
                "search_version": runtime.config.search.version,
                "prompt": prompt,
                "rank_backend": rank_backend,
                "rank_score": score_by_global[candidate_index],
                "display_index": display_index,
                "filter_relaxed": relaxed,
            },
        )
        artifacts.append(artifact)
    db.commit()
    return batch_id, artifacts, prompt


def fit_participant_preference_model(
    db: Session,
    runtime,
    participant: Participant,
) -> PairwisePreferenceModel | None:
    selections = list(
        db.scalars(
            select(Selection).where(
                Selection.participant_id == participant.participant_id
            )
        )
    )
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for selection in selections:
        winner = db.get(LatentImage, selection.selected_image_id)
        if winner is None:
            continue
        winner_latent = load_latent(winner)
        for image_id in json_loads(selection.shown_image_ids, []):
            if image_id == selection.selected_image_id:
                continue
            loser = db.get(LatentImage, image_id)
            if loser is not None:
                pairs.append((winner_latent, load_latent(loser)))
    if not pairs:
        return None
    model = PairwisePreferenceModel(runtime.projector).fit(pairs)
    path = (
        Path(runtime.config.paths.output_dir)
        / participant.participant_id
        / "preference_model.joblib"
    )
    model.save(path)
    return model


def generate_filtered_prior_candidates(
    runtime,
    target_gender: str | None,
    target_face_region: str | None,
    target_age_appearance: str | None,
    count: int,
    seed: int,
) -> list[EvaluatedCandidate]:
    pool_size = max(count * runtime.config.experiment.candidate_pool_multiplier, count)
    pool_size = _region_pool_size(
        runtime,
        target_face_region,
        target_age_appearance,
        pool_size,
        required_count=count,
    )
    accepted: list[EvaluatedCandidate] = []
    for attempt in range(int(runtime.config.filters.max_filter_attempts)):
        latents = runtime.generator.sample_prior(
            pool_size,
            seed + attempt * 15485863,
        )
        evaluated = evaluate_candidates(runtime, latents)
        accepted.extend(
            candidate
            for candidate in evaluated
            if candidate.quality_accepted
            and candidate.face_detected
            and _adult_accepted(runtime, candidate)
            and _portrait_accepted(runtime, candidate)
            and runtime.gender_controller.accepts(
                _gender_estimate(candidate),
                target_gender,
            )
        )
        if len(accepted) >= count:
            break
    if len(accepted) < count:
        raise RuntimeError(
            f"Only {len(accepted)} hard-valid gender/age faces were generated "
            f"for required={count}."
        )
    accepted.sort(
        key=lambda candidate: _demographic_preference_score(
            candidate,
            target_face_region,
            target_age_appearance,
        ),
        reverse=True,
    )
    return accepted[:count]


def evaluate_candidates(runtime, latents: np.ndarray) -> list[EvaluatedCandidate]:
    images = runtime.generator.decode(latents)
    generator_name = runtime.generator.generator_name
    shared_probabilities = (
        None
        if generator_name.startswith("demo")
        else runtime.clip_ranker.classify_groups(
            images,
            (
                runtime.gender_controller.prompts,
                runtime.adult_controller.prompts,
                runtime.portrait_controller.prompts,
                runtime.face_region_controller.prompts,
                runtime.age_appearance_controller.prompts,
            ),
        )
    )
    if shared_probabilities is None:
        gender_estimates = runtime.gender_controller.estimate(
            images,
            latents=latents,
            generator_name=generator_name,
        )
        adult_estimates = runtime.adult_controller.estimate(
            images,
            generator_name=generator_name,
        )
        portrait_estimates = runtime.portrait_controller.estimate(
            images,
            generator_name=generator_name,
        )
        face_region_estimates = runtime.face_region_controller.estimate(
            images,
            generator_name=generator_name,
        )
        age_appearance_estimates = runtime.age_appearance_controller.estimate(
            images,
            generator_name=generator_name,
        )
    else:
        gender_estimates = runtime.gender_controller.from_probabilities(
            shared_probabilities[0]
        )
        adult_estimates = runtime.adult_controller.from_probabilities(
            shared_probabilities[1]
        )
        portrait_estimates = runtime.portrait_controller.from_probabilities(
            shared_probabilities[2]
        )
        face_region_estimates = runtime.face_region_controller.from_probabilities(
            shared_probabilities[3]
        )
        age_appearance_estimates = (
            runtime.age_appearance_controller.from_probabilities(
                shared_probabilities[4]
            )
        )
    results: list[EvaluatedCandidate] = []
    for latent, image, gender, adult, portrait, face_region, age_appearance in zip(
        latents,
        images,
        gender_estimates,
        adult_estimates,
        portrait_estimates,
        face_region_estimates,
        age_appearance_estimates,
    ):
        quality = runtime.quality_filter.evaluate(
            image,
            require_face_detection=not runtime.generator.generator_name.startswith(
                "demo"
            ),
        )
        results.append(
            EvaluatedCandidate(
                latent=latent.astype(np.float32),
                image=image,
                gender_label=gender.label,
                gender_confidence=gender.confidence,
                gender_backend=gender.backend,
                is_adult=adult.is_adult,
                adult_confidence=adult.confidence,
                adult_backend=adult.backend,
                is_clean_portrait=portrait.is_clean_portrait,
                portrait_confidence=portrait.confidence,
                portrait_backend=portrait.backend,
                face_region_label=face_region.label,
                east_asian_confidence=face_region.east_asian_confidence,
                face_region_backend=face_region.backend,
                age_appearance_label=age_appearance.label,
                twenties_confidence=age_appearance.twenties_confidence,
                twenties_thirties_confidence=(
                    age_appearance.twenties_thirties_confidence
                ),
                age_appearance_backend=age_appearance.backend,
                quality_score=quality.score,
                quality_accepted=quality.accepted,
                face_detected=quality.face_detected,
                quality_backend=quality.backend,
            )
        )
    return results


def accepted_candidate_indices(
    runtime,
    evaluated: list[EvaluatedCandidate],
    target_gender: str | None,
    required: int,
) -> tuple[list[int], bool]:
    strict = [
        index
        for index, candidate in enumerate(evaluated)
        if candidate.quality_accepted
        and candidate.face_detected
        and _adult_accepted(runtime, candidate)
        and _portrait_accepted(runtime, candidate)
        and runtime.gender_controller.accepts(
            _gender_estimate(candidate),
            target_gender,
        )
    ]
    if len(strict) < required:
        raise RuntimeError(
            "Not enough generated faces passed the face, adult, portrait, and "
            "gender filters."
        )
    return strict, False


def strict_candidate_indices(
    runtime,
    evaluated: list[EvaluatedCandidate],
    target_gender: str | None,
) -> list[int]:
    return [
        index
        for index, candidate in enumerate(evaluated)
        if candidate.quality_accepted
        and candidate.face_detected
        and _adult_accepted(runtime, candidate)
        and _portrait_accepted(runtime, candidate)
        and runtime.gender_controller.accepts(
            _gender_estimate(candidate),
            target_gender,
        )
    ]


def unique_candidate_indices(
    features: np.ndarray,
    candidate_indices: list[int],
    threshold: float,
) -> list[int]:
    selected: list[int] = []
    for index in candidate_indices:
        if all(
            float(np.linalg.norm(features[index] - features[existing])) >= threshold
            for existing in selected
        ):
            selected.append(index)
    return selected


def region_boosted_candidate_indices(
    runtime,
    evaluated: list[EvaluatedCandidate],
    candidate_indices: list[int],
    required: int,
    preference: str | None,
    age_preference: str | None,
    preserve_indices: list[int],
    fallback_order: list[int] | None = None,
) -> list[int]:
    ordered_candidates = list(dict.fromkeys(candidate_indices))
    preserved = [
        index for index in preserve_indices if index in ordered_candidates
    ]
    fill_order = list(
        dict.fromkeys((fallback_order or []) + ordered_candidates)
    )
    region_enabled = preference not in {None, "", "unrestricted"}
    age_enabled = age_preference not in {None, "", "any_adult"}
    if not region_enabled and not age_enabled:
        return (preserved + [
            index for index in fill_order if index not in preserved
        ])[:required]

    selected = list(preserved)
    region_target = 0
    if region_enabled:
        region_fraction = (
            runtime.config.filters.east_asian_boost_fraction
            if preference == "east_asian_boost"
            else runtime.config.filters.balanced_east_asian_fraction
        )
        region_target = min(
            required,
            int(np.ceil(required * float(region_fraction))),
        )
    age_target = 0
    if age_enabled:
        age_fraction = (
            runtime.config.filters.twenties_boost_fraction
            if age_preference == "twenties_boost"
            else runtime.config.filters.twenties_thirties_fraction
        )
        age_target = min(
            required,
            int(np.ceil(required * float(age_fraction))),
        )

    remaining = [
        index for index in ordered_candidates if index not in selected
    ]
    while remaining and len(selected) < required:
        region_count = sum(
            evaluated[index].face_region_label == "east_asian"
            for index in selected
        )
        age_count = sum(
            _matches_age_preference(evaluated[index], age_preference)
            for index in selected
        )
        region_deficit = region_enabled and region_count < region_target
        age_deficit = age_enabled and age_count < age_target
        if not region_deficit and not age_deficit:
            break
        best = max(
            remaining,
            key=lambda index: (
                (
                    2.0
                    * (
                        evaluated[index].face_region_label == "east_asian"
                    )
                    + evaluated[index].east_asian_confidence
                )
                if region_deficit
                else 0.0
            )
            + (
                (
                    2.0 * _matches_age_preference(
                        evaluated[index],
                        age_preference,
                    )
                    + _age_preference_score(
                        evaluated[index],
                        age_preference,
                    )
                )
                if age_deficit
                else 0.0
            ),
        )
        selected.append(best)
        remaining.remove(best)

    for index in fill_order:
        if len(selected) >= required:
            break
        if index not in selected:
            selected.append(index)
    if len(selected) < required:
        raise RuntimeError(
            f"Only {len(selected)} region-balanced candidates were available "
            f"for required={required}."
        )
    return selected


def _region_pool_size(
    runtime,
    preference: str | None,
    age_preference: str | None,
    base_size: int,
    required_count: int,
) -> int:
    if runtime.generator.generator_name.startswith("demo"):
        return max(base_size, required_count * 4)
    if (
        preference in {None, "", "unrestricted"}
        and age_preference in {None, "", "any_adult"}
    ):
        return base_size
    configured_minimum = int(
        runtime.config.filters.east_asian_candidate_pool_minimum
    )
    minimum = min(
        configured_minimum,
        max(128, required_count * 64),
    )
    if preference == "balanced":
        minimum = max(64, minimum // 2)
    return max(base_size, minimum)


def _matches_age_preference(
    candidate: EvaluatedCandidate,
    age_preference: str | None,
) -> bool:
    if age_preference == "twenties_boost":
        return candidate.age_appearance_label == "20s"
    if age_preference == "twenties_thirties":
        return candidate.age_appearance_label in {"20s", "30s"}
    return True


def _age_preference_score(
    candidate: EvaluatedCandidate,
    age_preference: str | None,
) -> float:
    if age_preference == "twenties_boost":
        return candidate.twenties_confidence
    if age_preference == "twenties_thirties":
        return candidate.twenties_thirties_confidence
    return 0.0


def _demographic_preference_score(
    candidate: EvaluatedCandidate,
    face_region_preference: str | None,
    age_preference: str | None,
) -> float:
    score = candidate.quality_score * 0.05
    if face_region_preference not in {None, "", "unrestricted"}:
        score += (
            2.0 * (candidate.face_region_label == "east_asian")
            + candidate.east_asian_confidence
        )
    if age_preference not in {None, "", "any_adult"}:
        score += (
            2.0 * _matches_age_preference(candidate, age_preference)
            + _age_preference_score(candidate, age_preference)
        )
    return float(score)


def persist_candidate(
    db: Session,
    runtime,
    participant: Participant,
    candidate: EvaluatedCandidate,
    stage_type: str,
    condition_type: str,
    seed: int,
    block_id: str | None,
    round_id: int | None,
    batch_id: str | None,
    metadata: dict,
) -> LatentImage:
    image_id = str(uuid.uuid4())
    condition_segment = condition_type.replace("=", "_").replace(" ", "_")
    round_segment = f"round_{round_id:03d}" if round_id is not None else "shared"
    artifact_dir = (
        Path(runtime.config.paths.cache_dir)
        / participant.participant_id
        / stage_type
        / condition_segment
        / round_segment
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = latent_fingerprint(candidate.latent)
    latent_path = artifact_dir / f"{fingerprint}.npy"
    image_path = artifact_dir / f"{fingerprint}.png"
    latent_payload = array_to_npy_bytes(candidate.latent)
    image_payload = image_to_png_bytes(candidate.image)
    if not latent_path.exists():
        latent_path.write_bytes(latent_payload)
    if not image_path.exists():
        image_path.write_bytes(image_payload)

    score_metadata = {
        **metadata,
        "gender_backend": candidate.gender_backend,
        "adult_only_check": {
            "is_adult": candidate.is_adult,
            "confidence": candidate.adult_confidence,
            "backend": candidate.adult_backend,
        },
        "clean_portrait_check": {
            "accepted": candidate.is_clean_portrait,
            "confidence": candidate.portrait_confidence,
            "backend": candidate.portrait_backend,
        },
        "face_region_check": {
            "label": candidate.face_region_label,
            "east_asian_confidence": candidate.east_asian_confidence,
            "backend": candidate.face_region_backend,
        },
        "age_appearance_check": {
            "label": candidate.age_appearance_label,
            "twenties_confidence": candidate.twenties_confidence,
            "twenties_thirties_confidence": (
                candidate.twenties_thirties_confidence
            ),
            "backend": candidate.age_appearance_backend,
        },
        "face_quality_backend": candidate.quality_backend,
        "face_detected": candidate.face_detected,
        "latent_fingerprint": fingerprint,
    }
    artifact = LatentImage(
        image_id=image_id,
        participant_id=participant.participant_id,
        block_id=block_id,
        round_id=round_id,
        stage_type=stage_type,
        batch_id=batch_id,
        latent_path=str(latent_path),
        latent_data=(
            latent_payload if persist_artifacts_in_db(runtime.config) else None
        ),
        image_path=str(image_path),
        image_data=(
            image_payload if persist_artifacts_in_db(runtime.config) else None
        ),
        generator_type=runtime.generator.generator_name,
        generator_seed=int(seed),
        gender_target=participant.preferred_target_gender,
        gender_pred=candidate.gender_label,
        gender_confidence=candidate.gender_confidence,
        face_quality_score=candidate.quality_score,
        condition_type=condition_type,
        score_metadata_json=json_dumps(score_metadata),
    )
    db.add(artifact)
    db.flush()
    return artifact


def image_url(artifact: LatentImage, cache_dir: str) -> str:
    del cache_dir
    return f"/generated/{artifact.image_id}.png"


def _gender_estimate(candidate: EvaluatedCandidate):
    from core.gender_control import GenderEstimate

    return GenderEstimate(
        label=candidate.gender_label,
        confidence=candidate.gender_confidence,
        backend=candidate.gender_backend,
    )


def _adult_accepted(runtime, candidate: EvaluatedCandidate) -> bool:
    from core.adult_control import AdultEstimate

    return runtime.adult_controller.accepts(
        AdultEstimate(
            is_adult=candidate.is_adult,
            confidence=candidate.adult_confidence,
            backend=candidate.adult_backend,
        )
    )


def _portrait_accepted(runtime, candidate: EvaluatedCandidate) -> bool:
    from core.portrait_control import PortraitEstimate

    return runtime.portrait_controller.accepts(
        PortraitEstimate(
            is_clean_portrait=candidate.is_clean_portrait,
            confidence=candidate.portrait_confidence,
            backend=candidate.portrait_backend,
        )
    )
