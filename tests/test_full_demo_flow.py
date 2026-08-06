from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.base import get_session, reset_database_for_tests
from app.main import create_app
from app.models import (
    ExperimentBlock,
    FaceRating,
    FinalSurvey,
    LatentImage,
    Participant,
    ProfileChip,
    RecommendationEvaluation,
    Selection,
)
from app.settings import load_config


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
                "honest": "yes",
            },
        )

        for block_index in range(4):
            instruction = client.get(
                f"/experiment1/instructions?block={block_index}"
            )
            assert instruction.status_code == 200
            assert "Entropy Query" in instruction.text
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
            response = client.post(
                "/experiment1/round",
                data={
                    "block_index": str(block_index),
                    "round_id": "1",
                    "selected_image_id": selected_image_id,
                    "preference_rating": "8",
                    "difficulty": "3",
                    "reaction_time_sec": "1.25",
                },
                follow_redirects=False,
            )
            assert response.status_code == 303

        for _ in range(2):
            answer_page = client.get("/answer-key")
            assert answer_page.status_code == 200
            response = client.post(
                "/answer-key",
                data={
                    "image_id": _hidden_value(answer_page.text, "image_id"),
                    "rating": "8",
                },
                follow_redirects=False,
            )
            assert response.status_code == 303

        profile = client.get("/profile")
        assert profile.status_code == 200
        client.post(
            "/profile",
            data={
                "category::전반적 스타일/무드": "지적인",
                "category::머리 스타일": "웨이브",
            },
        )

        for _ in range(4):
            recommendation = client.get("/experiment2")
            assert recommendation.status_code == 200
            response = client.post(
                "/experiment2",
                data={
                    "batch_id": _hidden_value(recommendation.text, "batch_id"),
                    "selected_image_id": _first_image_id(recommendation.text),
                    "contains_ideal_type": "yes",
                    "rating": "7",
                    "reaction_time_sec": "2.5",
                },
                follow_redirects=False,
            )
            assert response.status_code == 303

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
        assert db.scalar(select(func.count(ExperimentBlock.block_id))) == 4
        assert db.scalar(select(func.count(Selection.id))) == 4
        assert db.scalar(select(func.count(RecommendationEvaluation.id))) == 4
        assert db.scalar(select(func.count(FaceRating.id))) == 10
        assert db.scalar(select(func.count(ProfileChip.id))) == 2
        assert db.scalar(select(func.count(FinalSurvey.id))) == 1
        assert db.scalar(select(func.count(LatentImage.image_id))) >= 18
        round_directory = (
            Path(first_block.strategy_state_path).parent / "round_01"
        )
        assert (round_directory / "query_points.npy").exists()
        assert (round_directory / "query_center.npy").exists()
        assert (round_directory / "posterior_mean.npy").exists()
        assert (round_directory / "posterior_covariance.npy").exists()
        assert (round_directory / "map_estimate.npy").exists()
        assert (round_directory / "entropy_metrics.json").exists()
        assert (round_directory / "observed_choice.json").exists()
        assert len(
            list((round_directory / "decoded_images").glob("*.png"))
        ) == first_block.m_value
