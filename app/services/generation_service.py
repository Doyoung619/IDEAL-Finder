from __future__ import annotations

import json
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
    ProfileChip,
    Selection,
)
from app.services.participant_service import m_order
from app.services.profile_service import build_profile_prompt
from core.latent_sampler import farthest_point_sampling
from core.preference_model import PairwisePreferenceModel
from core.query_strategy import ProposalBatch
from core.utils import latent_fingerprint
from experiments.design import stable_seed
from experiments.logging_utils import json_dumps, json_loads


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
        "east_asian_only",
        participant.preferred_age_appearance,
        count=8,
        seed=seed,
    )
    chosen = max(
        accepted,
        key=lambda item: _demographic_preference_score(
            item,
            "east_asian_only",
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
    if strategy_mode is None or strategy_parameters is None:
        raise ValueError("Algorithm selection is required before creating a block")

    baseline = ensure_participant_baseline(db, runtime, participant)
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
    initial_seed = stable_seed(
        f"{participant.participant_id}:benchmark-initial",
        participant.base_seed,
    )
    block = ExperimentBlock(
        block_id=block_id,
        participant_id=participant.participant_id,
        sequence_index=sequence_index,
        m_value=m_value,
        strategy_mode=strategy_mode,
        initial_state_id=latent_fingerprint(baseline),
        initial_seed=initial_seed,
        mu_path=str(mu_path),
        sigma=float(_initial_strategy_scale(strategy_mode, strategy_parameters)),
        strategy_state_path=str(state_path),
        strategy_parameters_json=json_dumps(strategy_parameters),
    )
    db.add(block)
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

    runtime.ensure_ready()
    strategy_parameters = json_loads(block.strategy_parameters_json or "{}")
    strategy = runtime.strategy_for(block.strategy_mode, strategy_parameters)
    center = np.load(block.mu_path)
    m_value = block.m_value
    line_modes = {"mvlq", "demographic_constrained_mvlq"}
    direct_modes = {
        "mvlq",
        "demographic_constrained_mvlq",
        "fast_direct",
        "heuristic",
        "axis_pair",
        "banditbo",
        "sequential_gallery",
    }
    if block.strategy_mode in line_modes:
        pool_size = m_value
    elif block.strategy_mode in direct_modes or (
        block.strategy_mode == "deepiec_mutation" and round_id > 1
    ):
        pool_size = max(
            int(runtime.config.experiment.direct_candidate_pool_minimum),
            m_value
            * int(runtime.config.experiment.direct_candidate_pool_multiplier),
        )
    else:
        pool_size = max(
            m_value * runtime.config.experiment.candidate_pool_multiplier,
            runtime.config.experiment.candidate_pool_size,
            4 if block.strategy_mode == "cmaes" else m_value,
        )
    needs_large_region_pool = block.strategy_mode in {
        "random",
        "linear_ucb",
        "simplebo",
        "qeubo",
        "trcb",
        "cmaes",
    } or (block.strategy_mode == "deepiec_mutation" and round_id == 1)
    has_demographic_boost = (
        participant.preferred_face_region != "unrestricted"
        or participant.preferred_age_appearance != "any_adult"
    )
    if has_demographic_boost and needs_large_region_pool:
        pool_size = _region_pool_size(
            runtime,
            participant.preferred_face_region,
            participant.preferred_age_appearance,
            pool_size,
            required_count=m_value,
        )
    pool_size = max(
        pool_size,
        int(strategy_parameters.get("candidate_pool_size", 0)),
        int(strategy_parameters.get("bank_size", 0))
        if block.strategy_mode == "trcb"
        else 0,
    )
    seed = (
        block.initial_seed
        if round_id == 1
        else stable_seed(
            f"{block.block_id}:round:{round_id}",
            participant.base_seed,
        )
    )
    duplicate_distance = (
        0.0
        if block.strategy_mode in line_modes
        else float(
            strategy_parameters.get(
                "duplicate_distance",
                runtime.config.search.duplicate_distance,
            )
        )
    )
    proposal_attempts = 0
    current_pool_size = pool_size
    maximum_pool_size = max(pool_size, 512)
    maximum_attempts = (
        12
        if block.strategy_mode in line_modes
        else int(runtime.config.filters.max_filter_attempts)
    )
    query_scale = 1.0
    while True:
        proposal_seed = seed + proposal_attempts * 15485863
        query_scale = (
            max(0.01, 0.70**proposal_attempts)
            if block.strategy_mode in line_modes
            else block.sigma
        )
        proposal = _propose_block_population(
            strategy,
            block,
            center,
            current_pool_size,
            proposal_seed,
            round_id,
            m_value,
            query_scale=query_scale,
        )
        evaluated = evaluate_candidates(runtime, proposal.latents)
        if block.strategy_mode == "demographic_constrained_mvlq":
            selection_features = proposal.features
            strict_indices = list(range(len(proposal.latents)))
        else:
            selection_features = runtime.projector.transform(proposal.latents)
            strict_indices = strict_candidate_indices(
                runtime,
                evaluated,
                participant.preferred_target_gender,
            )
        candidate_indices = unique_candidate_indices(
            selection_features,
            strict_indices,
            threshold=duplicate_distance,
        )
        if len(candidate_indices) >= m_value:
            proposal = ProposalBatch(
                latents=proposal.latents,
                features=selection_features,
                backend=proposal.backend,
                roles=proposal.roles,
                metadata=proposal.metadata,
            )
            break
        if (
            proposal_attempts + 1 >= maximum_attempts
            or (
                block.strategy_mode not in line_modes
                and current_pool_size >= maximum_pool_size
            )
        ):
            proposal = ProposalBatch(
                latents=proposal.latents,
                features=selection_features,
                backend=proposal.backend,
                roles=proposal.roles,
                metadata=proposal.metadata,
            )
            break
        proposal_attempts += 1
        if block.strategy_mode not in line_modes:
            current_pool_size = min(maximum_pool_size, current_pool_size * 2)

    recovery_used = False
    if len(candidate_indices) < m_value and block.strategy_mode not in line_modes:
        proposal, evaluated, strict_indices = _append_filter_recovery_candidates(
            runtime,
            proposal,
            evaluated,
            center,
            seed + 32452843,
            m_value,
            participant.preferred_target_gender,
        )
        recovery_used = True
        candidate_indices = unique_candidate_indices(
            proposal.features,
            strict_indices,
            threshold=duplicate_distance,
        )

    effective_duplicate_distance = duplicate_distance
    if len(candidate_indices) < m_value and len(strict_indices) >= m_value:
        for relaxed_distance in (
            duplicate_distance * 0.5,
            duplicate_distance * 0.25,
            0.0,
        ):
            candidate_indices = unique_candidate_indices(
                proposal.features,
                strict_indices,
                threshold=relaxed_distance,
            )
            effective_duplicate_distance = relaxed_distance
            if len(candidate_indices) >= m_value:
                break

    if len(candidate_indices) < m_value:
        raise RuntimeError(
            f"{block.strategy_mode} automatic recovery found only "
            f"{len(candidate_indices)} hard-valid candidates for M={m_value}."
        )
    preserve_indices: list[int] = []
    if block.strategy_mode == "deepiec_mutation" and round_id > 1:
        if 0 not in candidate_indices:
            raise RuntimeError("The preserved DeepIEC elite failed hard filters")
        preserve_indices = [0]
    elif (
        0 in candidate_indices
        and (
            block.strategy_mode in {"heuristic", "banditbo"}
            or (
                proposal.roles
                and proposal.roles[0] in {"elite", "incumbent"}
            )
        )
    ):
        preserve_indices = [0]
    selected_indices = (
        candidate_indices[:m_value]
        if block.strategy_mode in line_modes
        else region_boosted_candidate_indices(
            runtime,
            evaluated,
            candidate_indices,
            required=m_value,
            preference=participant.preferred_face_region,
            age_preference=participant.preferred_age_appearance,
            preserve_indices=preserve_indices,
        )
    )
    relaxed = False

    artifacts: list[LatentImage] = []
    for display_index, candidate_index in enumerate(selected_indices):
        candidate = evaluated[candidate_index]
        metadata = {
            "search_version": runtime.config.search.version,
            "strategy_mode": block.strategy_mode,
            "strategy_parameters": strategy_parameters,
            "proposal_backend": proposal.backend,
            "proposal_feature": proposal.features[candidate_index].tolist(),
            "display_index": display_index,
            "filter_relaxed": relaxed,
            "proposal_role": (
                proposal.roles[candidate_index] if proposal.roles else None
            ),
            "search_scale": query_scale,
            "mvlq_query_scale": (
                query_scale if block.strategy_mode in line_modes else None
            ),
            "constrained_mvlq": proposal.metadata,
            "face_region_preference": participant.preferred_face_region,
            "age_appearance_preference": participant.preferred_age_appearance,
            "proposal_attempts": proposal_attempts + 1,
            "filter_recovery_used": recovery_used,
            "effective_duplicate_distance": effective_duplicate_distance,
        }
        artifact = persist_candidate(
            db=db,
            runtime=runtime,
            participant=participant,
            candidate=candidate,
            stage_type="experiment1",
            condition_type=f"M={m_value}",
            seed=seed + candidate_index,
            block_id=block.block_id,
            round_id=round_id,
            batch_id=None,
            metadata=metadata,
        )
        artifacts.append(artifact)
    db.commit()
    return artifacts


def update_block_after_selection(
    db: Session,
    runtime,
    block: ExperimentBlock,
    shown_image_ids: list[str],
    selected_image_id: str,
    round_id: int,
) -> None:
    runtime.ensure_ready()
    strategy_parameters = json_loads(block.strategy_parameters_json or "{}")
    strategy = runtime.strategy_for(block.strategy_mode, strategy_parameters)
    shown = [db.get(LatentImage, image_id) for image_id in shown_image_ids]
    selected = db.get(LatentImage, selected_image_id)
    if selected is None or any(image is None for image in shown):
        raise ValueError("Selection refers to unknown latent images")
    shown_latents = np.vstack([np.load(image.latent_path) for image in shown])
    winner_latent = np.load(selected.latent_path)
    center = np.load(block.mu_path)
    new_center, new_sigma = strategy.update(
        center=center,
        sigma=block.sigma,
        shown_latents=shown_latents,
        winner_latent=winner_latent,
        state_path=block.strategy_state_path,
    )
    if block.strategy_mode == "mvlq":
        participant = db.get(Participant, block.participant_id)
        if participant is None:
            raise ValueError("MVLQ block refers to an unknown participant")
        center_evaluation = evaluate_candidates(runtime, new_center[None, :])
        center_is_valid = bool(
            strict_candidate_indices(
                runtime,
                center_evaluation,
                participant.preferred_target_gender,
            )
        )
        if not center_is_valid:
            new_center = winner_latent.astype(np.float32)
            strategy.constrain_map(new_center, block.strategy_state_path)
    np.save(block.mu_path, new_center.astype(np.float32))
    block.sigma = (
        max(
            float(
                strategy_parameters.get(
                    "sigma_min",
                    runtime.config.search.minimum_mutation_scale,
                )
            ),
            float(
                strategy_parameters.get(
                    "sigma_init",
                    runtime.config.search.initial_mutation_scale,
                )
            )
            * float(
                strategy_parameters.get(
                    "sigma_decay",
                    runtime.config.search.mutation_decay,
                )
            )
            ** max(round_id - 1, 0),
        )
        if block.strategy_mode == "deepiec_mutation"
        else new_sigma
    )
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
        winner_latent = np.load(winner.latent_path)
        for image_id in json_loads(selection.shown_image_ids, []):
            if image_id == selection.selected_image_id:
                continue
            loser = db.get(LatentImage, image_id)
            if loser is not None:
                pairs.append((winner_latent, np.load(loser.latent_path)))
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
            and _east_asian_accepted(runtime, candidate)
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
            f"Only {len(accepted)} hard-valid East Asian faces were generated "
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
        and _east_asian_accepted(runtime, candidate)
        and _adult_accepted(runtime, candidate)
        and _portrait_accepted(runtime, candidate)
        and runtime.gender_controller.accepts(
            _gender_estimate(candidate),
            target_gender,
        )
    ]
    if len(strict) < required:
        raise RuntimeError(
            "Not enough generated faces passed the hard East Asian, face, adult, "
            "portrait, and gender filters."
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
        and _east_asian_accepted(runtime, candidate)
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
    if not latent_path.exists():
        np.save(latent_path, candidate.latent.astype(np.float32))
    if not image_path.exists():
        candidate.image.save(image_path, format="PNG", optimize=True)

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
        image_path=str(image_path),
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
    relative = Path(artifact.image_path).resolve().relative_to(
        Path(cache_dir).resolve()
    )
    return f"/generated/{relative.as_posix()}"


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


def _east_asian_accepted(runtime, candidate: EvaluatedCandidate) -> bool:
    return (
        candidate.face_region_label == "east_asian"
        and candidate.east_asian_confidence
        >= float(runtime.config.filters.east_asian_confidence_threshold)
    )


def _initial_strategy_scale(strategy_mode: str, parameters: dict) -> float:
    if strategy_mode in {
        "deepiec_mutation",
        "heuristic",
        "fast_direct",
        "cmaes",
    }:
        return float(parameters.get("sigma_init", 0.5))
    if strategy_mode in {"axis_pair", "sequential_gallery"}:
        return float(parameters.get("step_init", 1.0))
    return 1.0


def _propose_block_population(
    strategy,
    block: ExperimentBlock,
    center: np.ndarray,
    count: int,
    seed: int,
    round_id: int,
    display_count: int,
    query_scale: float | None = None,
) -> ProposalBatch:
    if block.strategy_mode == "deepiec_mutation" and round_id == 1:
        return strategy.initial_population(count, seed)
    return strategy.propose(
        center=center,
        sigma=block.sigma if query_scale is None else query_scale,
        count=count,
        seed=seed,
        state_path=block.strategy_state_path,
        round_id=round_id,
        display_count=display_count,
    )


def _append_filter_recovery_candidates(
    runtime,
    proposal: ProposalBatch,
    evaluated: list[EvaluatedCandidate],
    center: np.ndarray,
    seed: int,
    required: int,
    target_gender: str | None,
) -> tuple[ProposalBatch, list[EvaluatedCandidate], list[int]]:
    recovery_parameters = {
        "latent_dim": min(8, runtime.config.search.pca_dimensions),
        "sigma_init": 0.12,
        "sigma_min": 0.03,
        "sigma_max": 0.20,
        "sigma_shrink": 0.70,
        "sigma_expand": 1.02,
    }
    recovery_strategy = runtime.strategy_for(
        "fast_direct",
        recovery_parameters,
    )
    combined_latents = np.asarray(proposal.latents, dtype=np.float32)
    combined_roles = list(
        proposal.roles or ["primary_proposal"] * len(combined_latents)
    )
    combined_evaluated = list(evaluated)
    for attempt in range(2):
        recovery_count = max(32, required * 6) * (attempt + 1)
        recovery = recovery_strategy.propose(
            center=center,
            sigma=0.12 / (attempt + 1),
            count=recovery_count,
            seed=seed + attempt * 49979687,
            round_id=1,
            display_count=required,
        )
        recovery_evaluated = evaluate_candidates(runtime, recovery.latents)
        combined_latents = np.vstack(
            [combined_latents, recovery.latents]
        ).astype(np.float32)
        combined_roles.extend(
            [f"filter_recovery_{attempt + 1}"] * len(recovery.latents)
        )
        combined_evaluated.extend(recovery_evaluated)
        combined_features = runtime.projector.transform(combined_latents)
        strict_indices = strict_candidate_indices(
            runtime,
            combined_evaluated,
            target_gender,
        )
        if len(strict_indices) >= required:
            return (
                ProposalBatch(
                    latents=combined_latents,
                    features=combined_features,
                    backend=f"{proposal.backend}+filter_recovery",
                    roles=combined_roles,
                ),
                combined_evaluated,
                strict_indices,
            )
    combined_features = runtime.projector.transform(combined_latents)
    return (
        ProposalBatch(
            latents=combined_latents,
            features=combined_features,
            backend=f"{proposal.backend}+filter_recovery",
            roles=combined_roles,
        ),
        combined_evaluated,
        strict_candidate_indices(
            runtime,
            combined_evaluated,
            target_gender,
        ),
    )
