from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import (
    ExperimentBlock,
    FaceRating,
    FinalSurvey,
    FinalRefinementEvaluation,
    LatentImage,
    Participant,
    PersonaInitialization,
    ProfileChip,
    RecommendationEvaluation,
    Selection,
)
from app.routes.helpers import current_participant, template_context
from app.services.generation_service import (
    find_block,
    generate_experiment_round,
    get_or_create_block,
    get_or_generate_answer_key_image,
    get_or_generate_recommendation_set,
    image_url,
    update_block_after_selection,
)
from app.services.final_evaluation_service import (
    prepare_final_artifacts,
    save_final_evaluation,
)
from core.algorithm_catalog import (
    algorithm_catalog,
    catalog_by_key,
)
from app.services.participant_service import (
    create_participant,
    m_order,
    recommendation_order,
    strategy_for_block,
)
from app.services.profile_service import PROFILE_CATEGORIES
from app.services.persona_service import (
    PERSONA_CATEGORIES,
    PersonaValidationError,
    answers_from_form,
    build_persona_prompt_bundle,
    complete_candidate_batch,
    confirm_persona_initialization,
    get_or_create_candidate_batch,
    get_persona_profile,
    latest_selected_batch,
    save_persona_profile,
)
from experiments.logging_utils import exit_screen, json_dumps, json_loads


router = APIRouter()

CONDITION_LABELS = {
    "none": "조건 A · 추가 정보 없음",
    "basic_info": "조건 B · 기본 정보",
    "click_profile": "조건 C · 클릭 선호 프로필",
    "selection_history": "조건 D · 선택 이력",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url=url, status_code=303)


def persona_prerequisite_url(db: Session, participant: Participant) -> str | None:
    if db.get(PersonaInitialization, participant.participant_id) is not None:
        return None
    if get_persona_profile(db, participant.participant_id) is None:
        return "/persona"
    if latest_selected_batch(db, participant.participant_id) is not None:
        return "/persona/confirm"
    return "/persona/candidates"


def initialized_destination(db: Session, participant: Participant) -> str:
    if participant.status == "completed":
        return "/complete"
    evaluation = db.scalar(
        select(FinalRefinementEvaluation).where(
            FinalRefinementEvaluation.participant_id == participant.participant_id
        )
    )
    if evaluation is not None:
        return "/survey"
    blocks = list(
        db.scalars(
            select(ExperimentBlock)
            .where(ExperimentBlock.participant_id == participant.participant_id)
            .order_by(ExperimentBlock.sequence_index)
        )
    )
    completed = sum(block.completed_at is not None for block in blocks)
    if completed >= len(m_order(participant)):
        return "/final-evaluation"
    return f"/experiment1/instructions?block={completed}"


@router.get("/")
def consent_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        request,
        "consent.html",
        template_context(request),
    )


@router.post("/consent")
async def submit_consent(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    if form.get("consent") != "yes":
        return request.app.state.templates.TemplateResponse(
            request,
            "consent.html",
            template_context(request, error="연구 참여 동의가 필요합니다."),
            status_code=422,
        )
    participant = create_participant(db, request.app.state.config)
    request.session.clear()
    request.session["participant_id"] = participant.participant_id
    return redirect("/basic-info")


@router.get("/basic-info")
def basic_info_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    return request.app.state.templates.TemplateResponse(
        request,
        "basic_info.html",
        template_context(
            request,
            participant,
            "basic_info",
            db,
        ),
    )


@router.post("/basic-info")
async def submit_basic_info(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    form = await request.form()
    required = [
        "age_band",
        "gender",
        "preferred_target_gender",
    ]
    if (
        any(not form.get(field) for field in required)
        or form.get("honest") != "yes"
        or form.get("preferred_target_gender") not in {"female", "male"}
    ):
        return request.app.state.templates.TemplateResponse(
            request,
            "basic_info.html",
            template_context(
                request,
                participant,
                error="모든 항목에 응답하고 직관적 참여 확인란을 선택해 주세요.",
            ),
            status_code=422,
        )
    participant.age_band = str(form["age_band"])
    participant.gender = str(form["gender"])
    participant.preferred_target_gender = str(form["preferred_target_gender"])
    participant.preferred_face_region = "unrestricted"
    participant.preferred_age_appearance = "twenties_boost"
    participant.honest_participation = True
    participant.status = "basic_info_complete"
    db.commit()
    exit_screen(db, participant.participant_id, "basic_info")
    return redirect("/persona")


@router.get("/persona")
def persona_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if not participant.preferred_target_gender:
        return redirect("/basic-info")
    if db.get(PersonaInitialization, participant.participant_id) is not None:
        return redirect(initialized_destination(db, participant))
    if latest_selected_batch(db, participant.participant_id) is not None:
        return redirect("/persona/confirm")
    profile = get_persona_profile(db, participant.participant_id)
    selected_values: set[str] = set()
    priorities: set[str] = set()
    if profile is not None:
        responses = json_loads(profile.responses_json, {})
        selected_values = {value for values in responses.values() for value in values}
        priorities = set(json_loads(profile.priorities_json, []))
    return request.app.state.templates.TemplateResponse(
        request,
        "persona_questionnaire.html",
        template_context(
            request,
            participant,
            "persona_questionnaire",
            db,
            categories=PERSONA_CATEGORIES,
            selected_values=selected_values,
            priorities=priorities,
        ),
    )


@router.post("/persona")
async def submit_persona(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if db.get(PersonaInitialization, participant.participant_id) is not None:
        return redirect(initialized_destination(db, participant))
    form = await request.form()
    answers, priorities = answers_from_form(form)
    try:
        bundle = build_persona_prompt_bundle(
            str(participant.preferred_target_gender),
            answers,
            priorities,
            priority_multiplier=float(request.app.state.config.persona.priority_multiplier),
        )
    except PersonaValidationError as exc:
        return request.app.state.templates.TemplateResponse(
            request,
            "persona_questionnaire.html",
            template_context(
                request,
                participant,
                categories=PERSONA_CATEGORIES,
                selected_values={value for values in answers.values() for value in values},
                priorities=set(priorities),
                error=str(exc),
            ),
            status_code=422,
        )
    save_persona_profile(
        db, participant, bundle, request.app.state.config.paths.output_dir
    )
    exit_screen(db, participant.participant_id, "persona_questionnaire")
    return redirect("/persona/candidates")


@router.get("/persona/candidates")
def persona_candidates_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if db.get(PersonaInitialization, participant.participant_id) is not None:
        return redirect(initialized_destination(db, participant))
    profile = get_persona_profile(db, participant.participant_id)
    if profile is None:
        return redirect("/persona")
    try:
        batch, artifacts = get_or_create_candidate_batch(
            db, request.app.state.runtime, participant, profile
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        return request.app.state.templates.TemplateResponse(
            request,
            "persona_candidates.html",
            template_context(
                request,
                participant,
                "persona_candidates_error",
                db,
                profile=profile,
                batch=None,
                cards=[],
                can_show_more=False,
                error=str(exc),
            ),
            status_code=503,
        )
    cards = [
        {
            "id": artifact.image_id,
            "url": image_url(artifact, request.app.state.config.paths.cache_dir),
        }
        for artifact in artifacts
    ]
    return request.app.state.templates.TemplateResponse(
        request,
        "persona_candidates.html",
        template_context(
            request,
            participant,
            f"persona_candidates_{batch.page_index}",
            db,
            profile=profile,
            batch=batch,
            cards=cards,
            can_show_more=(
                batch.page_index < int(request.app.state.config.persona.max_candidate_pages)
            ),
        ),
    )


@router.post("/persona/candidates")
async def submit_persona_candidates(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    form = await request.form()
    action = str(form.get("action", "selected"))
    try:
        complete_candidate_batch(
            db,
            participant,
            str(form.get("batch_id", "")),
            action,
            str(form.get("selected_image_id", "")) or None,
            float(form.get("reaction_time_sec", 0) or 0),
        )
    except (PersonaValidationError, ValueError):
        return redirect("/persona/candidates")
    if action == "edit_persona":
        return redirect("/persona")
    if action == "show_more":
        return redirect("/persona/candidates")
    return redirect("/persona/confirm")


@router.get("/persona/confirm")
def persona_confirm_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if db.get(PersonaInitialization, participant.participant_id) is not None:
        return redirect(initialized_destination(db, participant))
    batch = latest_selected_batch(db, participant.participant_id)
    profile = get_persona_profile(db, participant.participant_id)
    if profile is None:
        return redirect("/persona")
    if batch is None or batch.selected_image_id is None:
        return redirect("/persona/candidates")
    artifact = db.get(LatentImage, batch.selected_image_id)
    if artifact is None:
        return redirect("/persona/candidates")
    return request.app.state.templates.TemplateResponse(
        request,
        "persona_confirm.html",
        template_context(
            request,
            participant,
            "persona_confirm",
            db,
            profile=profile,
            batch=batch,
            image={
                "id": artifact.image_id,
                "url": image_url(artifact, request.app.state.config.paths.cache_dir),
            },
        ),
    )


@router.post("/persona/confirm")
async def submit_persona_confirm(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if db.get(PersonaInitialization, participant.participant_id) is not None:
        return redirect(initialized_destination(db, participant))
    batch = latest_selected_batch(db, participant.participant_id)
    if batch is None:
        return redirect("/persona/candidates")
    form = await request.form()
    try:
        confirm_persona_initialization(
            db,
            request.app.state.runtime,
            participant,
            batch,
            int(form.get("initial_rating", 0)),
            int(form.get("selection_confidence", 0)),
        )
    except (PersonaValidationError, ValueError):
        return redirect("/persona/confirm")
    exit_screen(db, participant.participant_id, "persona_confirm")
    return redirect("/experiment1/instructions?block=0")


@router.get("/experiment1/instructions")
def experiment1_instructions(
    request: Request,
    block: int = 0,
    db: Session = Depends(get_db),
):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    prerequisite = persona_prerequisite_url(db, participant)
    if request.app.state.config.persona.required and prerequisite:
        return redirect(prerequisite)
    order = m_order(participant)
    if block >= len(order):
        return redirect("/final-evaluation")
    experiment_block = find_block(db, participant, block)
    scheduled_mode = strategy_for_block(request.app.state.config, participant, block)
    algorithm = catalog_by_key(request.app.state.config)[scheduled_mode]
    selected_algorithm = (
        catalog_by_key(request.app.state.config).get(experiment_block.strategy_mode)
        if experiment_block
        else algorithm
    )
    return request.app.state.templates.TemplateResponse(
        request,
        "experiment1_instruction.html",
        template_context(
            request,
            participant,
            f"experiment1_instruction_{block}",
            db,
            block=experiment_block,
            m_value=order[block],
            block_number=block + 1,
            total_blocks=len(order),
            rounds=request.app.state.config.experiment.rounds_per_m,
            algorithm=algorithm,
            selected_algorithm=selected_algorithm,
            error=None,
        ),
    )


@router.post("/experiment1/start")
async def start_experiment1(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    prerequisite = persona_prerequisite_url(db, participant)
    if request.app.state.config.persona.required and prerequisite:
        return redirect(prerequisite)
    form = await request.form()
    block_index = int(form.get("block_index", 0))
    order = m_order(participant)
    if block_index < 0 or block_index >= len(order):
        return redirect("/final-evaluation")
    existing = find_block(db, participant, block_index)
    if existing is None:
        strategy_parameters = {
            "condition_gender": (
                participant.preferred_target_gender
                if participant.preferred_target_gender in {"female", "male"}
                else request.app.state.config.demographic.gender
            ),
            "condition_races": list(
                request.app.state.config.demographic.race_targets
            ),
        }
        get_or_create_block(
            db,
            request.app.state.runtime,
            participant,
            block_index,
            strategy_mode=strategy_for_block(
                request.app.state.config, participant, block_index
            ),
            strategy_parameters=strategy_parameters,
        )
    exit_screen(
        db,
        participant.participant_id,
        f"experiment1_instruction_{block_index}",
    )
    return redirect(f"/experiment1/round?block={block_index}")


@router.get("/experiment1/round")
def experiment1_round_page(
    request: Request,
    block: int,
    db: Session = Depends(get_db),
):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    prerequisite = persona_prerequisite_url(db, participant)
    if request.app.state.config.persona.required and prerequisite:
        return redirect(prerequisite)
    order = m_order(participant)
    if block >= len(order):
        return redirect("/final-evaluation")
    experiment_block = find_block(db, participant, block)
    if experiment_block is None:
        return redirect(f"/experiment1/instructions?block={block}")
    completed_rounds = db.scalar(
        select(func.count(Selection.id)).where(
            Selection.block_id == experiment_block.block_id
        )
    ) or 0
    total_rounds = request.app.state.config.experiment.rounds_per_m
    if completed_rounds >= total_rounds:
        if experiment_block.completed_at is None:
            experiment_block.completed_at = utc_now()
            db.commit()
        next_block = block + 1
        if next_block < len(order):
            return redirect(f"/experiment1/instructions?block={next_block}")
        return redirect("/final-evaluation")

    round_id = completed_rounds + 1
    try:
        artifacts = generate_experiment_round(
            db,
            request.app.state.runtime,
            participant,
            experiment_block,
            round_id,
        )
    except RuntimeError:
        algorithm = catalog_by_key(request.app.state.config).get(
            experiment_block.strategy_mode
        )
        return request.app.state.templates.TemplateResponse(
            request,
            "experiment1_instruction.html",
            template_context(
                request,
                participant,
                f"experiment1_recovery_{block}_{round_id}",
                db,
                block=experiment_block,
                m_value=experiment_block.m_value,
                block_number=block + 1,
                total_blocks=len(order),
                rounds=total_rounds,
                algorithm=algorithm_catalog(request.app.state.config)[0],
                selected_algorithm=algorithm,
                error=(
                    "유효 얼굴 자동 보충이 완료되지 않았습니다. "
                    "잠시 후 다시 시작을 누르면 새 후보로 재시도합니다."
                ),
            ),
            status_code=503,
        )
    cards = [
        {
            "id": artifact.image_id,
            "url": image_url(artifact, request.app.state.config.paths.cache_dir),
        }
        for artifact in artifacts
    ]
    return request.app.state.templates.TemplateResponse(
        request,
        "experiment1_round.html",
        template_context(
            request,
            participant,
            f"experiment1_block_{block}_round_{round_id}",
            db,
            cards=cards,
            block=experiment_block,
            block_index=block,
            round_id=round_id,
            total_rounds=total_rounds,
            strategy_label=(
                catalog_by_key(request.app.state.config)[
                    experiment_block.strategy_mode
                ].label
            ),
        ),
    )


@router.post("/experiment1/round")
async def submit_experiment1_round(
    request: Request,
    db: Session = Depends(get_db),
):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    prerequisite = persona_prerequisite_url(db, participant)
    if request.app.state.config.persona.required and prerequisite:
        return redirect(prerequisite)
    form = await request.form()
    block_index = int(form["block_index"])
    round_id = int(form["round_id"])
    selected_image_id = str(form.get("selected_image_id", ""))
    preference_rating = int(form.get("preference_rating", 0))
    difficulty = int(form.get("difficulty", 0))
    reaction_time = max(0.0, float(form.get("reaction_time_sec", 0)))
    block = find_block(db, participant, block_index)
    if block is None:
        return redirect(f"/experiment1/instructions?block={block_index}")
    existing = db.scalar(
        select(Selection).where(
            Selection.block_id == block.block_id,
            Selection.round_id == round_id,
        )
    )
    if existing is not None:
        return redirect(f"/experiment1/round?block={block_index}")

    artifacts = list(
        db.scalars(
            select(LatentImage).where(
                LatentImage.block_id == block.block_id,
                LatentImage.round_id == round_id,
                LatentImage.stage_type == "experiment1",
            )
        )
    )
    shown_ids = [artifact.image_id for artifact in artifacts]
    if (
        selected_image_id not in shown_ids
        or not 1 <= preference_rating <= 10
        or not 1 <= difficulty <= 7
    ):
        return redirect(f"/experiment1/round?block={block_index}")

    selection = Selection(
        participant_id=participant.participant_id,
        block_id=block.block_id,
        round_id=round_id,
        m_value=block.m_value,
        shown_image_ids=json_dumps(shown_ids),
        selected_image_id=selected_image_id,
        reaction_time_sec=reaction_time,
        difficulty_rating=difficulty,
    )
    db.add(selection)
    db.add(
        FaceRating(
            participant_id=participant.participant_id,
            stage_type="experiment1_selection",
            image_id=selected_image_id,
            rating_1_10=preference_rating,
            batch_id=f"{block.block_id}-round-{round_id:03d}",
            condition_type=f"M={block.m_value}",
        )
    )
    db.commit()
    update_block_after_selection(
        db,
        request.app.state.runtime,
        block,
        shown_ids,
        selected_image_id,
        round_id,
        preference_rating=preference_rating,
        difficulty_rating=difficulty,
    )
    exit_screen(
        db,
        participant.participant_id,
        f"experiment1_block_{block_index}_round_{round_id}",
    )
    return redirect(f"/experiment1/round?block={block_index}")


@router.get("/final-evaluation")
def final_evaluation_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    prerequisite = persona_prerequisite_url(db, participant)
    if request.app.state.config.persona.required and prerequisite:
        return redirect(prerequisite)
    existing = db.scalar(
        select(FinalRefinementEvaluation).where(
            FinalRefinementEvaluation.participant_id == participant.participant_id
        )
    )
    if existing is not None:
        return redirect("/survey")
    completed_blocks = db.scalar(
        select(func.count(ExperimentBlock.block_id)).where(
            ExperimentBlock.participant_id == participant.participant_id,
            ExperimentBlock.completed_at.is_not(None),
        )
    ) or 0
    if completed_blocks < len(m_order(participant)):
        return redirect(f"/experiment1/instructions?block={completed_blocks}")
    try:
        initial, final_map, last_winner, display = prepare_final_artifacts(
            db, request.app.state.runtime, participant
        )
    except RuntimeError:
        return redirect("/experiment1/instructions?block=0")
    cards = [
        {
            "id": artifact.image_id,
            "url": image_url(artifact, request.app.state.config.paths.cache_dir),
        }
        for artifact in display
    ]
    return request.app.state.templates.TemplateResponse(
        request,
        "final_evaluation.html",
        template_context(
            request,
            participant,
            "final_evaluation",
            db,
            cards=cards,
            initial_id=initial.image_id,
            final_id=final_map.image_id,
            last_winner_id=last_winner.image_id if last_winner else None,
        ),
    )


@router.post("/final-evaluation")
async def submit_final_evaluation(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    try:
        initial, final_map, last_winner, display = prepare_final_artifacts(
            db, request.app.state.runtime, participant
        )
    except RuntimeError:
        return redirect("/experiment1/instructions?block=0")
    form = await request.form()
    try:
        save_final_evaluation(
            db,
            participant,
            initial,
            final_map,
            last_winner,
            display,
            str(form.get("preferred_image_id", "")),
            int(form.get("initial_rating", 0)),
            int(form.get("final_rating", 0)),
            int(form.get("perceived_improvement", 0)),
            int(form.get("final_match", 0)),
            float(form.get("reaction_time_sec", 0) or 0),
        )
    except (ValueError, TypeError):
        return redirect("/final-evaluation")
    exit_screen(db, participant.participant_id, "final_evaluation")
    return redirect("/survey")


@router.get("/answer-key")
def answer_key_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if not request.app.state.config.evaluation.answer_key_enabled:
        return redirect("/final-evaluation")
    completed = db.scalar(
        select(func.count(FaceRating.id)).where(
            FaceRating.participant_id == participant.participant_id,
            FaceRating.stage_type == "answer_key",
        )
    ) or 0
    total = request.app.state.config.experiment.answer_key_faces
    if completed >= total:
        return redirect("/profile")
    index = completed + 1
    artifact = get_or_generate_answer_key_image(
        db,
        request.app.state.runtime,
        participant,
        index,
    )
    return request.app.state.templates.TemplateResponse(
        request,
        "answer_key.html",
        template_context(
            request,
            participant,
            f"answer_key_{index}",
            db,
            image={
                "id": artifact.image_id,
                "url": image_url(
                    artifact,
                    request.app.state.config.paths.cache_dir,
                ),
            },
            index=index,
            total=total,
        ),
    )


@router.post("/answer-key")
async def submit_answer_key(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if not request.app.state.config.evaluation.answer_key_enabled:
        return redirect("/final-evaluation")
    form = await request.form()
    image_id = str(form.get("image_id", ""))
    rating = int(form.get("rating", 0))
    artifact = db.get(LatentImage, image_id)
    existing = db.scalar(
        select(FaceRating).where(
            FaceRating.participant_id == participant.participant_id,
            FaceRating.stage_type == "answer_key",
            FaceRating.image_id == image_id,
        )
    )
    if (
        artifact is not None
        and artifact.participant_id == participant.participant_id
        and artifact.stage_type == "answer_key"
        and existing is None
        and 1 <= rating <= 10
    ):
        db.add(
            FaceRating(
                participant_id=participant.participant_id,
                stage_type="answer_key",
                image_id=image_id,
                rating_1_10=rating,
                batch_id=artifact.batch_id,
                condition_type="answer_key",
            )
        )
        db.commit()
        exit_screen(
            db,
            participant.participant_id,
            f"answer_key_{artifact.round_id}",
        )
    return redirect("/answer-key")


@router.get("/profile")
def profile_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if request.app.state.config.persona.enabled:
        return redirect("/persona")
    selected = list(
        db.scalars(
            select(ProfileChip).where(
                ProfileChip.participant_id == participant.participant_id
            )
        )
    )
    selected_values = {chip.selected_option for chip in selected}
    return request.app.state.templates.TemplateResponse(
        request,
        "profile.html",
        template_context(
            request,
            participant,
            "profile",
            db,
            categories=PROFILE_CATEGORIES,
            selected_values=selected_values,
        ),
    )


@router.post("/profile")
async def submit_profile(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if request.app.state.config.persona.enabled:
        return redirect("/persona")
    form = await request.form()
    selections: dict[str, list[str]] = {}
    for key, value in form.multi_items():
        if key.startswith("category::"):
            category = key.removeprefix("category::")
            selections.setdefault(category, []).append(str(value))
    invalid = any(
        category not in PROFILE_CATEGORIES
        or len(values) > 2
        or any(value not in PROFILE_CATEGORIES[category] for value in values)
        for category, values in selections.items()
    )
    if invalid or not any(selections.values()):
        return request.app.state.templates.TemplateResponse(
            request,
            "profile.html",
            template_context(
                request,
                participant,
                categories=PROFILE_CATEGORIES,
                selected_values={
                    value for values in selections.values() for value in values
                },
                error="각 항목은 최대 2개이며, 전체에서 하나 이상 선택해 주세요.",
            ),
            status_code=422,
        )
    db.execute(
        delete(ProfileChip).where(
            ProfileChip.participant_id == participant.participant_id
        )
    )
    for category, values in selections.items():
        for value in values:
            db.add(
                ProfileChip(
                    participant_id=participant.participant_id,
                    category=category,
                    selected_option=value,
                )
            )
    db.commit()
    exit_screen(db, participant.participant_id, "profile")
    return redirect("/experiment2")


@router.get("/experiment2")
def experiment2_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if request.app.state.config.persona.enabled:
        return redirect("/final-evaluation")
    conditions = recommendation_order(participant)
    batches_per_condition = (
        request.app.state.config.experiment.recommendation_batches_per_condition
    )
    next_item: tuple[int, str, int] | None = None
    completed_sets = 0
    for condition_index, condition in enumerate(conditions):
        for batch_index in range(1, batches_per_condition + 1):
            batch_id = (
                f"{participant.participant_id}-recommendation-"
                f"{condition}-{batch_index:02d}"
            )
            completed = db.scalar(
                select(RecommendationEvaluation).where(
                    RecommendationEvaluation.batch_id == batch_id
                )
            )
            if completed is None and next_item is None:
                next_item = (condition_index, condition, batch_index)
            elif completed is not None:
                completed_sets += 1
    if next_item is None:
        return redirect("/survey")

    condition_index, condition, batch_index = next_item
    batch_id, artifacts, prompt = get_or_generate_recommendation_set(
        db,
        request.app.state.runtime,
        participant,
        condition,
        batch_index,
    )
    cards = [
        {
            "id": artifact.image_id,
            "url": image_url(artifact, request.app.state.config.paths.cache_dir),
        }
        for artifact in artifacts
    ]
    total_sets = len(conditions) * batches_per_condition
    return request.app.state.templates.TemplateResponse(
        request,
        "recommendation.html",
        template_context(
            request,
            participant,
            f"recommendation_{condition}_{batch_index}",
            db,
            condition=condition,
            condition_label=CONDITION_LABELS[condition],
            condition_number=condition_index + 1,
            total_conditions=len(conditions),
            batch_index=batch_index,
            batches_per_condition=batches_per_condition,
            batch_id=batch_id,
            cards=cards,
            prompt=prompt,
            progress_current=completed_sets + 1,
            progress_total=total_sets,
        ),
    )


@router.post("/experiment2")
async def submit_experiment2(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if request.app.state.config.persona.enabled:
        return redirect("/final-evaluation")
    form = await request.form()
    batch_id = str(form.get("batch_id", ""))
    selected_image_id = str(form.get("selected_image_id", ""))
    contains_value = str(form.get("contains_ideal_type", ""))
    rating = int(form.get("rating", 0))
    reaction_time = max(0.0, float(form.get("reaction_time_sec", 0)))
    existing = db.scalar(
        select(RecommendationEvaluation).where(
            RecommendationEvaluation.batch_id == batch_id
        )
    )
    artifacts = list(
        db.scalars(
            select(LatentImage).where(
                LatentImage.batch_id == batch_id,
                LatentImage.participant_id == participant.participant_id,
            )
        )
    )
    shown_ids = [artifact.image_id for artifact in artifacts]
    if (
        existing is None
        and selected_image_id in shown_ids
        and contains_value in {"yes", "no"}
        and 1 <= rating <= 10
    ):
        condition = artifacts[0].condition_type
        db.add(
            RecommendationEvaluation(
                participant_id=participant.participant_id,
                batch_id=batch_id,
                condition_type=condition,
                shown_image_ids=json_dumps(shown_ids),
                contains_ideal_type=contains_value == "yes",
                selected_image_id=selected_image_id,
                selected_rating_1_10=rating,
                reaction_time_sec=reaction_time,
            )
        )
        db.add(
            FaceRating(
                participant_id=participant.participant_id,
                stage_type="recommendation_eval",
                image_id=selected_image_id,
                rating_1_10=rating,
                batch_id=batch_id,
                condition_type=condition,
            )
        )
        db.commit()
        exit_screen(
            db,
            participant.participant_id,
            f"recommendation_{condition}_{artifacts[0].round_id}",
        )
    return redirect("/experiment2")


@router.get("/survey")
def survey_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if request.app.state.config.evaluation.final_refinement_enabled:
        evaluation = db.scalar(
            select(FinalRefinementEvaluation).where(
                FinalRefinementEvaluation.participant_id == participant.participant_id
            )
        )
        if evaluation is None:
            return redirect("/final-evaluation")
    existing = db.scalar(
        select(FinalSurvey).where(
            FinalSurvey.participant_id == participant.participant_id
        )
    )
    if existing is not None:
        return redirect("/complete")
    return request.app.state.templates.TemplateResponse(
        request,
        "survey.html",
        template_context(request, participant, "final_survey", db),
    )


@router.post("/survey")
async def submit_survey(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    if request.app.state.config.evaluation.final_refinement_enabled:
        evaluation = db.scalar(
            select(FinalRefinementEvaluation).where(
                FinalRefinementEvaluation.participant_id == participant.participant_id
            )
        )
        if evaluation is None:
            return redirect("/final-evaluation")
    form = await request.form()
    values = [int(form.get(f"q{index}", 0)) for index in range(1, 5)]
    if not all(1 <= value <= 7 for value in values):
        return request.app.state.templates.TemplateResponse(
            request,
            "survey.html",
            template_context(
                request,
                participant,
                error="모든 문항에 1~7점으로 응답해 주세요.",
            ),
            status_code=422,
        )
    existing = db.scalar(
        select(FinalSurvey).where(
            FinalSurvey.participant_id == participant.participant_id
        )
    )
    if existing is None:
        db.add(
            FinalSurvey(
                participant_id=participant.participant_id,
                q1=values[0],
                q2=values[1],
                q3=values[2],
                q4=values[3],
                free_text=str(form.get("free_text", "")).strip() or None,
            )
        )
        participant.status = "completed"
        participant.completed_at = utc_now()
        db.commit()
        exit_screen(db, participant.participant_id, "final_survey")
    return redirect("/complete")


@router.get("/complete")
def complete_page(request: Request, db: Session = Depends(get_db)):
    participant = current_participant(request, db)
    if participant is None:
        return redirect("/")
    return request.app.state.templates.TemplateResponse(
        request,
        "complete.html",
        template_context(
            request,
            participant,
            "complete",
            db,
            submitted_at=participant.completed_at or utc_now(),
        ),
    )


@router.post("/new-participant")
def new_participant(request: Request):
    request.session.clear()
    return redirect("/")
