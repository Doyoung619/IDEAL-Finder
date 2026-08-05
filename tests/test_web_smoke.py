from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.base import get_session, reset_database_for_tests
from app.main import create_app
from app.models import ExperimentBlock, LatentImage, Participant
from app.settings import load_config


def test_demo_flow_reaches_first_generated_round(tmp_path):
    config = load_config(demo_override=True)
    config.database._values["url"] = f"sqlite:///{tmp_path / 'web.sqlite3'}"
    config.paths._values["data_dir"] = str(tmp_path / "data")
    config.paths._values["cache_dir"] = str(tmp_path / "cache")
    config.paths._values["output_dir"] = str(tmp_path / "outputs")
    config.paths._values["model_dir"] = str(tmp_path / "models")
    config.search._values["pca_dimensions"] = 4
    config.search._values["pca_fit_samples"] = 16
    config.experiment._values["candidate_pool_size"] = 64
    config.experiment._values["candidate_pool_multiplier"] = 4
    config.generator._values["output_resolution"] = 32
    reset_database_for_tests(config.database.url)
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "동의하고 시작하기" in response.text

        response = client.post(
            "/consent",
            data={"consent": "yes"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        response = client.post(
            "/basic-info",
            data={
                "age_band": "20s",
                "gender": "female",
                "preferred_target_gender": "male",
                "preferred_age_appearance": "twenties_boost",
                "dating_experience": "past",
                "image_selection_importance": "4",
                "honest": "yes",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        instruction = client.get(response.headers["location"])
        assert instruction.status_code == 200
        assert "Maximum-Variance Line Query" in instruction.text
        assert 'name="strategy_mode"' not in instruction.text
        block_index = 0
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
        assert "가장 마음에 드는 얼굴을 선택하세요" in round_page.text

    with get_session() as db:
        participant = db.scalar(select(Participant))
        block = db.scalar(select(ExperimentBlock))
        images = list(
            db.scalars(
                select(LatentImage).where(
                    LatentImage.participant_id == participant.participant_id
                )
            )
        )
        assert len(images) == block.m_value
        assert block.strategy_mode == "mvlq"
        assert participant.preferred_face_region == "east_asian_only"
        assert participant.preferred_age_appearance == "twenties_boost"
        assert block.strategy_parameters_json == "{}"
        assert all(image.latent_path.endswith(".npy") for image in images)
