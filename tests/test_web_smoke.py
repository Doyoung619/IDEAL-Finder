from __future__ import annotations

import re

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.base import get_session, reset_database_for_tests
from app.main import create_app
from app.models import ExperimentBlock, LatentImage, Participant, PersonaInitialization
from app.services.persona_service import PERSONA_CATEGORIES
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
        basic_info = client.get(response.headers["location"])
        assert basic_info.status_code == 200
        assert "연애 경험" not in basic_info.text
        assert "평소 사진 선택에서 외모가 얼마나 중요한가요?" not in basic_info.text
        assert "생성 얼굴 범위는 동아시아 인상으로 고정됩니다." not in basic_info.text
        response = client.post(
            "/basic-info",
            data={
                "age_band": "20s",
                "gender": "female",
                "preferred_target_gender": "male",
                "honest": "yes",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303

        persona = client.get(response.headers["location"])
        assert persona.status_code == 200
        assert "어떤 인상의 얼굴" in persona.text
        persona_data = {
            f"persona::{category.key}": "no_preference"
            for category in PERSONA_CATEGORIES
        }
        response = client.post("/persona", data=persona_data, follow_redirects=False)
        assert response.headers["location"] == "/persona/candidates"
        candidates = client.get(response.headers["location"])
        image_id = re.search(r'data-image-id="([^"]+)"', candidates.text).group(1)
        batch_id = re.search(r'name="batch_id" value="([^"]+)"', candidates.text).group(1)
        client.post(
            "/persona/candidates",
            data={
                "batch_id": batch_id,
                "selected_image_id": image_id,
                "action": "selected",
                "reaction_time_sec": "1.0",
            },
        )
        client.post(
            "/persona/confirm",
            data={"initial_rating": "7", "selection_confidence": "6"},
        )

        instruction = client.get("/experiment1/instructions?block=0")
        assert instruction.status_code == 200
        assert "Entropy Query" in instruction.text or "RC-MLQ (Ours)" in instruction.text
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
        image_match = re.search(r'<img src="([^"]+)"', round_page.text)
        assert image_match is not None
        image_response = client.get(image_match.group(1))
        assert image_response.status_code == 200
        assert image_response.headers["content-type"] == "image/png"

    with get_session() as db:
        participant = db.scalar(select(Participant))
        block = db.scalar(select(ExperimentBlock))
        images = list(
            db.scalars(
                select(LatentImage).where(
                    LatentImage.participant_id == participant.participant_id,
                    LatentImage.stage_type == "experiment1",
                )
            )
        )
        assert len(images) == block.m_value
        assert block.strategy_mode in {"entropy", "rc_mlq"}
        assert participant.preferred_face_region == "unrestricted"
        assert participant.preferred_age_appearance == "twenties_boost"
        assert db.get(PersonaInitialization, participant.participant_id) is not None
        assert '"condition_gender":"male"' in block.strategy_parameters_json
        assert all(image.latent_path.endswith(".npy") for image in images)
