from __future__ import annotations

import re
import pickle
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.base import get_session, reset_database_for_tests
from app.main import create_app
from app.models import (
    ExperimentBlock,
    ExperimentEvent,
    ExperimentRound,
    ExperimentSession,
    FaceRating,
    FinalRefinementEvaluation,
    FinalSurvey,
    LatentImage,
    Participant,
    PersonaCandidateBatch,
    PersonaInitialization,
    PersonaProfile,
    ProfileChip,
    RecommendationEvaluation,
    Selection,
)
from app.settings import load_config
from app.services.persona_service import PERSONA_CATEGORIES
from app.services.artifact_storage import load_latent


def _hidden_value(html: str, name: str) -> str:
    match = re.search(
        rf'<input[^>]+name="{re.escape(name)}"[^>]+value="([^"]+)"',
        html,
    )
    assert match is not None
    return match.group(1)


def _first_image_id(html: str) -> str:
    match = re.search(r'data-image-id="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def test_complete_demo_experiment(tmp_path):
    config = load_config(demo_override=True)
    config.database._values["url"] = f"sqlite:///{tmp_path / 'full.sqlite3'}"
    config.paths._values["data_dir"] = str(tmp_path / "data")
    config.paths._values["cache_dir"] = str(tmp_path / "cache")
    config.paths._values["output_dir"] = str(tmp_path / "outputs")
    config.paths._values["model_dir"] = str(tmp_path / "models")
    config.generator._values["output_resolution"] = 32
    config.search._values["pca_dimensions"] = 4
    config.search._values["pca_fit_samples"] = 16
    config.experiment._values["candidate_pool_size"] = 64
    config.experiment._values["candidate_pool_multiplier"] = 4
    config.experiment._values["answer_key_faces"] = 2
    config.experiment._values["recommendation_set_size"] = 2
    config.query_diversity._values.update(
        {
            "enabled": True,
            "image_similarity_threshold": -1.0,
            "max_retries": 1,
            "perturbation_initial": 0.10,
            "perturbation_max": 0.10,
        }
    )
    reset_database_for_tests(config.database.url)
    app = create_app(config)

    with TestClient(app) as client:
        client.post("/consent", data={"consent": "yes"})
        client.post(
            "/basic-info",
            data={
                "age_band": "30s",
                "gender": "male",
                "preferred_target_gender": "female",
                "preferred_age_appearance": "twenties_boost",
                "dating_experience": "current",
                "image_selection_importance": "3",
                "honest": "yes",
            },
        )

        persona_page = client.get("/persona")
        assert "선호 대상 · 여성" in persona_page.text
        assert "단발" in persona_page.text
        assert "묶은 머리" in persona_page.text

        persona_data = {
            f"persona::{category.key}": "no_preference"
            for category in PERSONA_CATEGORIES
        }
        client.post("/persona", data=persona_data)
        with get_session() as db:
            assert db.scalar(select(func.count(ExperimentSession.session_id))) == 0
        candidates = client.get("/persona/candidates")
        selected_persona_id = _first_image_id(candidates.text)
        client.post(
            "/persona/candidates",
            data={
                "batch_id": _hidden_value(candidates.text, "batch_id"),
                "selected_image_id": selected_persona_id,
                "action": "selected",
                "reaction_time_sec": "1.0",
            },
        )
        with get_session() as db:
            assert db.scalar(select(func.count(ExperimentSession.session_id))) == 0
        client.post(
            "/persona/confirm",
            data={"initial_rating": "8", "selection_confidence": "6"},
        )
        with get_session() as db:
            assert db.scalar(select(func.count(ExperimentSession.session_id))) == 1

        first_frontend_selected_id = None
        for block_index in range(6):
            instruction = client.get(
                f"/experiment1/instructions?block={block_index}"
            )
            assert instruction.status_code == 200
            assert (
                "Entropy Query" in instruction.text
                or "RC-MLQ (Ours)" in instruction.text
            )
            assert 'name="strategy_mode"' not in instruction.text
            start = client.post(
                "/experiment1/start",
                data={
                    "block_index": str(block_index),
                },
                follow_redirects=False,
            )
            assert start.status_code == 303
            round_page = client.get(f"/experiment1/round?block={block_index}")
            assert round_page.status_code == 200
            assert "data-elapsed" not in round_page.text
            assert 'name="preference_rating"' in round_page.text
            selected_image_id = _first_image_id(round_page.text)
            if block_index == 0:
                first_frontend_selected_id = selected_image_id
            submission = {
                "block_index": str(block_index),
                "round_id": "1",
                "selected_image_id": selected_image_id,
                "preference_rating": "8",
                "difficulty": "3",
                "reaction_time_sec": "1.25",
            }
            response = client.post(
                "/experiment1/round",
                data=submission,
                follow_redirects=False,
            )
            assert response.status_code == 303
            if block_index == 0:
                duplicate = client.post(
                    "/experiment1/round",
                    data=submission,
                    follow_redirects=False,
                )
                assert duplicate.status_code == 303
            client.get(f"/experiment1/round?block={block_index}")

        final_page = client.get("/final-evaluation")
        assert final_page.status_code == 200
        response = client.post(
            "/final-evaluation",
            data={
                "preferred_image_id": _first_image_id(final_page.text),
                "initial_rating": "8",
                "final_rating": "9",
                "perceived_improvement": "6",
                "final_match": "9",
                "reaction_time_sec": "2.5",
            },
            follow_redirects=False,
        )
        assert response.headers["location"] == "/survey"

        survey = client.get("/survey")
        assert survey.status_code == 200
        completion = client.post(
            "/survey",
            data={
                "q1": "6",
                "q2": "5",
                "q3": "6",
                "q4": "7",
                "free_text": "좋았습니다.",
            },
        )
        assert completion.status_code == 200
        assert "참여해 주셔서" in completion.text

    with get_session() as db:
        participant = db.scalar(select(Participant))
        first_block = db.scalar(
            select(ExperimentBlock).order_by(ExperimentBlock.sequence_index)
        )
        assert participant.status == "completed"
        assert db.scalar(select(func.count(ExperimentBlock.block_id))) == 6
        assert db.scalar(select(func.count(Selection.id))) == 6
        assert db.scalar(select(func.count(ExperimentRound.round_id))) == 6
        assert db.scalar(
            select(func.count(ExperimentRound.round_id)).where(
                ExperimentRound.status == "completed"
            )
        ) == 6
        experiment_session = db.scalar(select(ExperimentSession))
        assert experiment_session.status == "completed"
        assert experiment_session.completed_at is not None
        assert db.scalar(select(func.count(ExperimentEvent.event_id))) >= 20
        assert db.scalar(select(func.count(RecommendationEvaluation.id))) == 0
        assert db.scalar(select(func.count(FaceRating.id))) == 6
        assert db.scalar(select(func.count(ProfileChip.id))) == 0
        assert db.scalar(select(func.count(PersonaProfile.id))) == 1
        assert db.scalar(select(func.count(PersonaCandidateBatch.batch_id))) == 1
        assert db.scalar(select(func.count(PersonaInitialization.participant_id))) == 1
        assert db.scalar(select(func.count(FinalRefinementEvaluation.id))) == 1
        assert db.scalar(select(func.count(FinalSurvey.id))) == 1
        assert db.scalar(select(func.count(LatentImage.image_id))) >= 36
        initialization = db.get(PersonaInitialization, participant.participant_id)
        selected_theta = np.load(initialization.theta_path)
        blocks = list(
            db.scalars(select(ExperimentBlock).order_by(ExperimentBlock.sequence_index))
        )
        assert len({block.strategy_state_path for block in blocks}) == 6
        assert sorted(block.m_value for block in blocks) == [2, 2, 4, 4, 8, 8]
        for m_value in (2, 4, 8):
            assert {
                block.strategy_mode for block in blocks if block.m_value == m_value
            } == {"entropy", "rc_mlq"}
        for block in blocks:
            with Path(block.strategy_state_path).open("rb") as handle:
                state = pickle.load(handle)
            assert np.array_equal(state["mixture"]["local_mean"], selected_theta)
            assert state["mixture"]["global_weight"] == 0.45
            assert state["prior_covariance_scale"] == 0.35
        round_directory = (
            Path(first_block.strategy_state_path).parent / "round_01"
        )
        assert (round_directory / "query_points.npy").exists()
        assert (round_directory / "final_query_points.npy").exists()
        assert (round_directory / "query_center.npy").exists()
        assert (round_directory / "posterior_mean.npy").exists()
        assert (round_directory / "posterior_covariance.npy").exists()
        assert (round_directory / "map_estimate.npy").exists()
        assert (
            round_directory / f"{first_block.strategy_mode}_metrics.json"
        ).exists()
        assert (round_directory / "observed_choice.json").exists()
        assert len(
            list((round_directory / "decoded_images").glob("*.png"))
        ) == first_block.m_value

        original_theta = np.load(round_directory / "query_points.npy")
        final_theta = np.load(round_directory / "final_query_points.npy")
        assert not np.array_equal(original_theta, final_theta)
        selection = db.scalar(
            select(Selection).where(Selection.block_id == first_block.block_id)
        )
        experiment_round = db.scalar(
            select(ExperimentRound).where(
                ExperimentRound.block_id == first_block.block_id
            )
        )
        shown_ids = json.loads(selection.shown_image_ids)
        assert selection.selected_image_id == first_frontend_selected_id
        artifacts = [db.get(LatentImage, image_id) for image_id in shown_ids]
        prior = app.state.runtime.conditional_prior(
            json.loads(first_block.strategy_parameters_json)
        )
        persisted_theta = np.vstack(
            [prior.w_to_theta(load_latent(artifact)) for artifact in artifacts]
        )
        assert np.allclose(persisted_theta, final_theta, atol=1e-6)
        assert np.allclose(experiment_round.final_query_points, final_theta)
        assert experiment_round.generated_image_ids == shown_ids
        assert experiment_round.selected_image_id == first_frontend_selected_id
        for index, artifact in enumerate(artifacts):
            rerendered = app.state.runtime.generator.decode(
                np.atleast_2d(load_latent(artifact))
            )[0]
            displayed = Image.open(
                round_directory / "decoded_images" / f"query_{index + 1:02d}.png"
            ).convert("RGB")
            assert np.array_equal(np.asarray(rerendered), np.asarray(displayed))
        with Path(first_block.strategy_state_path).open("rb") as handle:
            updated_state = pickle.load(handle)
        assert np.allclose(updated_state["history"][0]["queries"], final_theta)
        assert updated_state["history"][0]["winner_index"] == shown_ids.index(
            first_frontend_selected_id
        )
