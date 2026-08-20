from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    LatentImage,
    Participant,
    PersonaCandidateBatch,
    PersonaInitialization,
    PersonaProfile,
)
from app.services.artifact_storage import (
    array_to_npy_bytes,
    persist_artifacts_in_db,
)
from experiments.design import stable_seed


logger = logging.getLogger("ideal_finder.persona")


PERSONA_SCHEMA_VERSION = "persona_v1"
FIXED_RACE = "unrestricted"
FIXED_AGE = "20_29"


@dataclass(frozen=True)
class PersonaOption:
    key: str
    label_ko: str
    prompt_en: str | None


@dataclass(frozen=True)
class PersonaCategory:
    key: str
    label_ko: str
    maximum: int
    options: tuple[PersonaOption, ...]
    mutually_exclusive: tuple[frozenset[str], ...] = ()

    def conflicts_for(self, option_key: str) -> tuple[str, ...]:
        conflicts = {
            other
            for group in self.mutually_exclusive
            if option_key in group
            for other in group
            if other != option_key
        }
        return tuple(sorted(conflicts))


@dataclass(frozen=True)
class PersonaPromptBundle:
    schema_version: str
    target_gender: str
    raw_profile: dict
    korean_summary: str
    base_prompt: str
    full_prompt: str
    category_prompts: tuple[str, ...]
    category_keys: tuple[str, ...]
    category_weights: tuple[float, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _option(key: str, label: str, prompt: str | None) -> PersonaOption:
    return PersonaOption(key, label, prompt)


NO_PREFERENCE = _option("no_preference", "상관없음 / 잘 모르겠음", None)

PERSONA_CATEGORIES: tuple[PersonaCategory, ...] = (
    PersonaCategory(
        "overall_mood",
        "전반적인 분위기",
        2,
        (
            _option("overall_gentle", "순한", "gentle and approachable appearance"),
            _option("overall_calm", "차분한", "calm appearance"),
            _option("bright_friendly", "밝고 친근한", "bright and friendly appearance"),
            _option("cute_adult", "귀여운 성인 느낌", "charming and cute adult appearance"),
            _option("chic", "시크한", "chic appearance"),
            _option("intellectual", "지적인", "intellectual appearance"),
            _option("overall_mature", "성숙한", "mature adult appearance"),
            _option("natural", "자연스러운", "natural appearance"),
            _option("overall_charismatic", "카리스마 있는", "charismatic appearance"),
            NO_PREFERENCE,
        ),
    ),
    PersonaCategory(
        "face_shape",
        "얼굴형",
        1,
        (
            _option("oval", "계란형", "oval face shape"),
            _option("round", "둥근형", "round face shape"),
            _option("long_face", "긴형", "long face shape"),
            _option("angular", "각진형", "angular face shape"),
            _option("heart", "하트형", "heart-shaped face"),
            _option("v_line", "갸름한 V라인", "slender V-shaped jawline"),
            NO_PREFERENCE,
        ),
    ),
    PersonaCategory(
        "hair_length",
        "머리 길이",
        1,
        (
            _option("short_hair", "숏컷", "short haircut"),
            _option("bob_hair", "단발", "bob haircut"),
            _option("medium_hair", "중간 길이", "medium-length hair"),
            _option("long_hair", "긴 머리", "long hair"),
            NO_PREFERENCE,
        ),
    ),
    PersonaCategory(
        "hair_style",
        "머리 스타일",
        2,
        (
            _option("straight_hair", "생머리", "straight hair"),
            _option("wavy_hair", "자연스러운 웨이브", "naturally wavy hair"),
            _option("bangs", "앞머리 있음", "with bangs"),
            _option("no_bangs", "앞머리 없음", "without bangs"),
            _option("tied_hair", "묶은 머리", "tied-back hair"),
            _option("neat_hair", "단정한 스타일", "neatly styled hair"),
            NO_PREFERENCE,
        ),
        (frozenset(("bangs", "no_bangs")),),
    ),
    PersonaCategory(
        "eyes",
        "눈의 특징",
        2,
        (
            _option("large_eyes", "큰 눈", "large eyes"),
            _option("small_eyes", "작은 눈", "small eyes"),
            _option("narrow_eyes", "가는 눈", "narrow eyes"),
            _option("defined_eyes", "또렷한 눈매", "clear defined eyes"),
            _option("soft_eyes", "부드러운 눈매", "soft gentle eyes"),
            _option("double_eyelids", "쌍꺼풀", "double eyelids"),
            _option("monolid", "무쌍", "monolid eyes"),
            NO_PREFERENCE,
        ),
        (
            frozenset(("large_eyes", "small_eyes")),
            frozenset(("double_eyelids", "monolid")),
        ),
    ),
    PersonaCategory(
        "nose_mouth",
        "코와 입",
        2,
        (
            _option("high_nose", "높은 콧대", "high nose bridge"),
            _option("small_nose", "작은 코", "small nose"),
            _option("full_lips", "도톰한 입술", "full lips"),
            _option("natural_lips", "자연스러운 입술", "natural lips"),
            _option("upturned_mouth", "입꼬리가 올라간 느낌", "slightly upturned mouth corners"),
            _option("pleasant_smile", "미소가 예쁜 느낌", "pleasant smile"),
            NO_PREFERENCE,
        ),
    ),
    PersonaCategory(
        "impression",
        "인상",
        2,
        (
            _option("gentle_impression", "순한 인상", "gentle impression"),
            _option("cool_impression", "차가운 인상", "cool reserved impression"),
            _option("mature_impression", "성숙한 인상", "mature adult impression"),
            _option("youthful_adult", "어려 보이는 성인 인상", "youthful adult appearance"),
            _option("cute_impression", "귀여운 인상", "charming adult impression"),
            _option("urban_impression", "도시적인 인상", "sophisticated urban impression"),
            _option("charismatic_impression", "카리스마 있는 인상", "charismatic impression"),
            NO_PREFERENCE,
        ),
    ),
    PersonaCategory(
        "expression",
        "표정",
        1,
        (
            _option("neutral_expression", "중립적인 표정", "neutral expression"),
            _option("slight_smile", "살짝 미소", "subtle smile"),
            _option("bright_smile", "밝은 미소", "bright natural smile"),
            _option("calm_expression", "차분한 표정", "calm expression"),
            NO_PREFERENCE,
        ),
    ),
)


MALE_HAIR_LENGTH_OPTIONS = (
    _option("short_hair", "짧은 머리", "short haircut"),
    _option("medium_hair", "보통 길이", "medium-length hair"),
    _option("long_hair", "장발", "long hair"),
    NO_PREFERENCE,
)

MALE_HAIR_STYLE_OPTIONS = (
    _option("straight_hair", "직모 / 생머리", "straight hair"),
    _option("wavy_hair", "자연스러운 웨이브", "naturally wavy hair"),
    _option("bangs", "앞머리 있음", "with bangs"),
    _option("no_bangs", "앞머리 없음", "without bangs"),
    _option("neat_hair", "단정한 스타일", "neatly styled hair"),
    NO_PREFERENCE,
)


def persona_categories_for_gender(target_gender: str) -> tuple[PersonaCategory, ...]:
    """Return the questionnaire options appropriate for the requested face."""
    if target_gender != "male":
        return PERSONA_CATEGORIES
    categories: list[PersonaCategory] = []
    for category in PERSONA_CATEGORIES:
        if category.key == "hair_length":
            category = replace(category, options=MALE_HAIR_LENGTH_OPTIONS)
        elif category.key == "hair_style":
            category = replace(category, options=MALE_HAIR_STYLE_OPTIONS)
        categories.append(category)
    return tuple(categories)

class PersonaValidationError(ValueError):
    pass


def validate_persona_profile(
    target_gender: str,
    answers: Mapping[str, Sequence[str]],
    priorities: Sequence[str],
) -> dict[str, list[str]]:
    if target_gender not in {"female", "male"}:
        raise PersonaValidationError("선호 대상 성별은 여성 또는 남성이어야 합니다.")
    categories = persona_categories_for_gender(target_gender)
    category_by_key = {category.key: category for category in categories}
    unknown_categories = set(answers) - set(category_by_key)
    if unknown_categories:
        raise PersonaValidationError("알 수 없는 persona 문항이 포함되어 있습니다.")

    normalized: dict[str, list[str]] = {}
    selected: set[str] = set()
    for category in categories:
        values = list(dict.fromkeys(str(value) for value in answers.get(category.key, ())))
        allowed = {option.key for option in category.options}
        if not values:
            raise PersonaValidationError(f"'{category.label_ko}' 문항에 답해주세요.")
        if set(values) - allowed:
            raise PersonaValidationError(f"'{category.label_ko}'에 알 수 없는 선택지가 있습니다.")
        if "no_preference" in values and len(values) != 1:
            raise PersonaValidationError("'상관없음'은 다른 항목과 함께 선택할 수 없습니다.")
        concrete = [value for value in values if value != "no_preference"]
        if len(concrete) > category.maximum:
            raise PersonaValidationError(
                f"'{category.label_ko}'은(는) 최대 {category.maximum}개까지 선택할 수 있습니다."
            )
        for conflict in category.mutually_exclusive:
            if conflict.issubset(concrete):
                raise PersonaValidationError(
                    f"'{category.label_ko}'에 서로 모순되는 항목이 포함되어 있습니다."
                )
        normalized[category.key] = values
        selected.update(concrete)

    normalized_priorities = list(dict.fromkeys(str(value) for value in priorities))
    if len(normalized_priorities) > 3:
        raise PersonaValidationError("중요한 특징은 최대 3개까지 선택할 수 있습니다.")
    if set(normalized_priorities) - selected:
        raise PersonaValidationError("중요한 특징은 앞에서 고른 항목 중에서만 선택할 수 있습니다.")
    normalized["priorities"] = normalized_priorities
    return normalized


def build_persona_prompt_bundle(
    target_gender: str,
    answers: Mapping[str, Sequence[str]],
    priorities: Sequence[str],
    priority_multiplier: float = 2.0,
    fixed_race: str = FIXED_RACE,
) -> PersonaPromptBundle:
    validated = validate_persona_profile(target_gender, answers, priorities)
    categories = persona_categories_for_gender(target_gender)
    option_by_key = {
        option.key: option
        for category in categories
        for option in category.options
        if option.key != "no_preference"
    }
    noun = "woman" if target_gender == "female" else "man"
    base_prompt = (
        "a high-quality realistic studio head-and-shoulders portrait photograph "
        f"of one adult {noun} aged 20 to 29, front-facing, natural "
        "skin texture, balanced lighting, plain neutral background, clearly visible face"
    )
    category_prompts: list[str] = []
    category_keys: list[str] = []
    category_phrases: list[str] = []
    korean_parts: list[str] = []
    raw_weights: list[float] = []
    priority_set = set(validated["priorities"])

    for category in categories:
        keys = validated[category.key]
        concrete = [key for key in keys if key != "no_preference"]
        if not concrete:
            korean_parts.append(f"{category.label_ko}: 상관없음")
            continue
        options = [option_by_key[key] for key in concrete]
        phrases = [option.prompt_en for option in options if option.prompt_en]
        labels = [option.label_ko for option in options]
        phrase = ", ".join(phrases)
        category_prompts.append(
            f"a realistic portrait of an adult {noun} aged 20 to 29 with {phrase}"
        )
        category_keys.append(category.key)
        category_phrases.extend(phrases)
        korean_parts.append(f"{category.label_ko}: {', '.join(labels)}")
        raw_weights.append(
            float(priority_multiplier)
            if priority_set.intersection(concrete)
            else 1.0
        )

    if category_phrases:
        full_prompt = f"{base_prompt}, with " + ", ".join(category_phrases)
    else:
        full_prompt = base_prompt
    weight_sum = sum(raw_weights)
    category_weights = tuple(
        weight / weight_sum for weight in raw_weights
    ) if raw_weights else ()
    raw_profile = {
        "schema_version": PERSONA_SCHEMA_VERSION,
        "target_gender": target_gender,
        "fixed_race": fixed_race,
        "fixed_age": FIXED_AGE,
        "answers": {
            category.key: validated[category.key]
            for category in categories
        },
        "priorities": validated["priorities"],
    }
    return PersonaPromptBundle(
        schema_version=PERSONA_SCHEMA_VERSION,
        target_gender=target_gender,
        raw_profile=raw_profile,
        korean_summary=" · ".join(korean_parts),
        base_prompt=base_prompt,
        full_prompt=full_prompt,
        category_prompts=tuple(category_prompts),
        category_keys=tuple(category_keys),
        category_weights=category_weights,
    )


def answers_from_form(
    form, target_gender: str
) -> tuple[dict[str, list[str]], list[str]]:
    answers = {
        category.key: []
        for category in persona_categories_for_gender(target_gender)
    }
    priorities: list[str] = []
    for key, value in form.multi_items():
        if key.startswith("persona::"):
            category = key.removeprefix("persona::")
            if category in answers:
                answers[category].append(str(value))
        elif key == "priority":
            priorities.append(str(value))
    return answers, priorities


def prompt_bundle_from_profile(profile: PersonaProfile) -> PersonaPromptBundle:
    values = json.loads(profile.prompt_bundle_json)
    return PersonaPromptBundle(
        schema_version=values["schema_version"],
        target_gender=values["target_gender"],
        raw_profile=values["raw_profile"],
        korean_summary=values["korean_summary"],
        base_prompt=values["base_prompt"],
        full_prompt=values["full_prompt"],
        category_prompts=tuple(values["category_prompts"]),
        category_keys=tuple(values["category_keys"]),
        category_weights=tuple(values["category_weights"]),
    )


def save_persona_profile(
    db: Session,
    participant: Participant,
    bundle: PersonaPromptBundle,
    output_root: str | Path,
) -> PersonaProfile:
    profile = db.scalar(
        select(PersonaProfile).where(
            PersonaProfile.participant_id == participant.participant_id
        )
    )
    now = datetime.now(timezone.utc)
    payload = bundle.as_dict()
    profile_changed = (
        profile is not None
        and profile.prompt_bundle_json
        != json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    if profile_changed:
        open_batches = list(
            db.scalars(
                select(PersonaCandidateBatch).where(
                    PersonaCandidateBatch.participant_id == participant.participant_id,
                    PersonaCandidateBatch.action.is_(None),
                )
            )
        )
        for batch in open_batches:
            batch.action = "edit_persona"
            batch.completed_at = now
    if profile is None:
        profile = PersonaProfile(
            participant_id=participant.participant_id,
            schema_version=bundle.schema_version,
            target_gender=bundle.target_gender,
            fixed_race=str(bundle.raw_profile["fixed_race"]),
            fixed_age_range=FIXED_AGE,
            responses_json="{}",
            priorities_json="[]",
            korean_summary=bundle.korean_summary,
            full_prompt_en=bundle.full_prompt,
            prompt_bundle_json="{}",
        )
        db.add(profile)
    profile.schema_version = bundle.schema_version
    profile.target_gender = bundle.target_gender
    profile.fixed_race = str(bundle.raw_profile["fixed_race"])
    profile.fixed_age_range = FIXED_AGE
    profile.responses_json = json.dumps(
        bundle.raw_profile["answers"], ensure_ascii=False, sort_keys=True
    )
    profile.priorities_json = json.dumps(
        bundle.raw_profile["priorities"], ensure_ascii=False
    )
    profile.korean_summary = bundle.korean_summary
    profile.full_prompt_en = bundle.full_prompt
    profile.prompt_bundle_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    profile.updated_at = now
    participant.status = "persona_complete"
    db.flush()
    artifact_dir = Path(output_root) / participant.participant_id / "persona"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "persona_profile.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    db.commit()
    return profile


def get_persona_profile(db: Session, participant_id: str) -> PersonaProfile | None:
    return db.scalar(
        select(PersonaProfile).where(PersonaProfile.participant_id == participant_id)
    )


def get_or_create_candidate_batch(
    db: Session,
    runtime,
    participant: Participant,
    profile: PersonaProfile,
) -> tuple[PersonaCandidateBatch, list[LatentImage]]:
    started = time.perf_counter()
    open_batch = db.scalar(
        select(PersonaCandidateBatch)
        .where(
            PersonaCandidateBatch.participant_id == participant.participant_id,
            PersonaCandidateBatch.action.is_(None),
        )
        .order_by(PersonaCandidateBatch.created_at.desc())
    )
    if open_batch is not None:
        ids = json.loads(open_batch.shown_image_ids_json)
        images = [db.get(LatentImage, image_id) for image_id in ids]
        return open_batch, [image for image in images if image is not None]

    previous = list(
        db.scalars(
            select(PersonaCandidateBatch)
            .where(
                PersonaCandidateBatch.participant_id == participant.participant_id,
                PersonaCandidateBatch.created_at >= profile.updated_at,
            )
            .order_by(PersonaCandidateBatch.created_at)
        )
    )
    page_index = len(previous) + 1
    if page_index > int(runtime.config.persona.max_candidate_pages):
        raise RuntimeError("Persona candidate page limit has been reached")
    excluded = {
        str(pool_id)
        for item in previous
        for pool_id in json.loads(item.shown_pool_ids_json)
    }
    pool = runtime.persona_pool(profile.target_gender)
    prompt_bundle = prompt_bundle_from_profile(profile)
    seed = stable_seed(
        f"{participant.participant_id}:persona-page:{page_index}",
        participant.base_seed,
    )
    retrieved = runtime.persona_retriever().retrieve(
        prompt_bundle, pool, excluded_pool_ids=excluded, seed=seed
    )
    batch_id = f"{participant.participant_id}-persona-{uuid.uuid4().hex[:12]}"
    image_ids: list[str] = []
    page_dir = (
        Path(runtime.config.paths.cache_dir)
        / participant.participant_id
        / "persona_candidates"
        / f"page_{page_index:02d}"
    )
    page_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[LatentImage] = []
    for display_index, item in enumerate(retrieved):
        pool_index = item.pool_index
        image_id = str(uuid.uuid4())
        image_path = page_dir / f"candidate_{display_index + 1:02d}.png"
        theta_path = page_dir / f"candidate_{display_index + 1:02d}_theta.npy"
        w_path = page_dir / f"candidate_{display_index + 1:02d}_w.npy"
        shutil.copy2(pool.image_path(pool_index), image_path)
        np.save(theta_path, pool.theta[pool_index].astype(np.float32))
        np.save(w_path, pool.w[pool_index].astype(np.float32))
        metadata = {
            "source": "persona_pool",
            "pool_id": item.pool_id,
            "pool_version": pool.metadata.get("pool_version", "unknown"),
            "theta_path": str(theta_path),
            "semantic_score": item.semantic_score,
            "semantic_rank": item.semantic_rank,
            "mmr_score": item.mmr_score,
            "display_index": display_index,
            "persona_schema_version": profile.schema_version,
            "clip_backend": runtime.clip_ranker.backend,
        }
        artifact = LatentImage(
            image_id=image_id,
            participant_id=participant.participant_id,
            block_id=None,
            round_id=page_index,
            stage_type="persona_candidate",
            batch_id=batch_id,
            latent_path=str(w_path),
            latent_data=(
                array_to_npy_bytes(pool.w[pool_index])
                if persist_artifacts_in_db(runtime.config)
                else None
            ),
            image_path=str(image_path),
            image_data=(
                image_path.read_bytes()
                if persist_artifacts_in_db(runtime.config)
                else None
            ),
            generator_type=runtime.generator.generator_name,
            generator_seed=int(pool.generator_seed[pool_index]),
            gender_target=profile.target_gender,
            gender_pred=profile.target_gender,
            gender_confidence=float(pool.gender_probability[pool_index]),
            face_quality_score=float(pool.quality_score[pool_index]),
            condition_type="persona_warm_start",
            score_metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        )
        db.add(artifact)
        artifacts.append(artifact)
        image_ids.append(image_id)
    pool_version = str(pool.metadata.get("pool_version", "unknown"))
    batch = PersonaCandidateBatch(
        batch_id=batch_id,
        participant_id=participant.participant_id,
        page_index=page_index,
        pool_version=pool_version,
        shown_pool_ids_json=json.dumps([item.pool_id for item in retrieved]),
        shown_image_ids_json=json.dumps(image_ids),
        semantic_scores_json=json.dumps([item.semantic_score for item in retrieved]),
        semantic_ranks_json=json.dumps([item.semantic_rank for item in retrieved]),
        mmr_scores_json=json.dumps([item.mmr_score for item in retrieved]),
        display_order_json=json.dumps(image_ids),
    )
    db.add(batch)
    participant.status = "persona_candidates_presented"
    db.commit()
    logger.info(
        "persona_latency participant_id=%s candidate_count=%s "
        "total_request_latency_ms=%.1f",
        participant.participant_id,
        len(artifacts),
        (time.perf_counter() - started) * 1000.0,
    )
    return batch, artifacts


def complete_candidate_batch(
    db: Session,
    participant: Participant,
    batch_id: str,
    action: str,
    selected_image_id: str | None,
    reaction_time_sec: float,
) -> PersonaCandidateBatch:
    if action not in {"selected", "show_more", "edit_persona"}:
        raise PersonaValidationError("알 수 없는 후보 선택 동작입니다.")
    batch = db.get(PersonaCandidateBatch, batch_id)
    if batch is None or batch.participant_id != participant.participant_id:
        raise PersonaValidationError("후보 묶음을 찾을 수 없습니다.")
    if batch.action is not None:
        return batch
    image_ids = json.loads(batch.shown_image_ids_json)
    pool_ids = json.loads(batch.shown_pool_ids_json)
    if action == "selected":
        if selected_image_id not in image_ids:
            raise PersonaValidationError("화면에 표시된 얼굴을 선택해 주세요.")
        selected_index = image_ids.index(selected_image_id)
        batch.selected_image_id = selected_image_id
        batch.selected_pool_id = pool_ids[selected_index]
    batch.action = action
    batch.reaction_time_sec = max(0.0, float(reaction_time_sec))
    batch.completed_at = datetime.now(timezone.utc)
    db.commit()
    return batch


def latest_selected_batch(
    db: Session, participant_id: str
) -> PersonaCandidateBatch | None:
    return db.scalar(
        select(PersonaCandidateBatch)
        .where(
            PersonaCandidateBatch.participant_id == participant_id,
            PersonaCandidateBatch.action == "selected",
        )
        .order_by(PersonaCandidateBatch.completed_at.desc())
    )


def confirm_persona_initialization(
    db: Session,
    runtime,
    participant: Participant,
    batch: PersonaCandidateBatch,
    initial_rating: int,
    selection_confidence: int,
) -> PersonaInitialization:
    existing = db.get(PersonaInitialization, participant.participant_id)
    if existing is not None:
        return existing
    if not 1 <= int(initial_rating) <= 10 or not 1 <= int(selection_confidence) <= 7:
        raise PersonaValidationError("평가 문항에 모두 응답해 주세요.")
    profile = get_persona_profile(db, participant.participant_id)
    selected = db.get(LatentImage, batch.selected_image_id)
    if profile is None or selected is None or batch.selected_pool_id is None:
        raise PersonaValidationError("확정할 persona 후보 정보가 없습니다.")
    metadata = json.loads(selected.score_metadata_json)
    source_theta = Path(metadata["theta_path"])
    source_w = Path(selected.latent_path)
    source_image = Path(selected.image_path)
    output_dir = Path(runtime.config.paths.output_dir) / participant.participant_id / "persona"
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_path = output_dir / "selected_theta.npy"
    w_path = output_dir / "selected_w.npy"
    image_path = output_dir / "selected_image.png"
    np.save(theta_path, np.asarray(np.load(source_theta), dtype=np.float32))
    np.save(w_path, np.asarray(np.load(source_w), dtype=np.float32))
    shutil.copy2(source_image, image_path)
    image_ids = json.loads(batch.shown_image_ids_json)
    selected_index = image_ids.index(selected.image_id)
    scores = json.loads(batch.semantic_scores_json)
    ranks = json.loads(batch.semantic_ranks_json)
    initialization = PersonaInitialization(
        participant_id=participant.participant_id,
        persona_profile_id=profile.id,
        candidate_batch_id=batch.batch_id,
        selected_pool_id=batch.selected_pool_id,
        selected_image_id=selected.image_id,
        selected_rank=int(ranks[selected_index]),
        semantic_score=float(scores[selected_index]),
        theta_path=str(theta_path),
        w_path=str(w_path),
        initial_rating_1_10=int(initial_rating),
        selection_confidence_1_7=int(selection_confidence),
        pool_version=batch.pool_version,
    )
    db.add(initialization)
    participant.baseline_latent_path = str(w_path)
    participant.status = "warm_start_selected"
    metadata_payload = {
        "source": "persona_pool",
        "pool_id": batch.selected_pool_id,
        "pool_version": batch.pool_version,
        "selected_image_id": selected.image_id,
        "selected_rank": int(ranks[selected_index]),
        "semantic_score": float(scores[selected_index]),
        "persona_schema_version": profile.schema_version,
        "prior_covariance_scale": float(runtime.config.persona.prior_covariance_scale),
    }
    (output_dir / "initialization_metadata.json").write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    db.commit()
    return initialization
