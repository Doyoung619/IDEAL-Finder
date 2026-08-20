from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.base import get_session, reset_database_for_tests
from app.main import create_app
from app.models import (
    ExperimentBlock,
    ExperimentRound,
    ExperimentSession,
    LatentImage,
    Selection,
)
from app.services.persona_service import PERSONA_CATEGORIES
from app.settings import load_config


def _hidden_value(html: str, name: str) -> str:
    match = re.search(
        rf'<input[^>]+name="{re.escape(name)}"[^>]+value="([^"]+)"', html
    )
    assert match is not None
    return match.group(1)


def _first_image_id(html: str) -> str:
    match = re.search(r'data-image-id="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def _complete_persona(client: TestClient) -> None:
    client.post("/consent", data={"consent": "yes"})
    client.post(
        "/basic-info",
        data={
            "age_band": "20s",
            "gender": "female",
            "preferred_target_gender": "male",
            "preferred_age_appearance": "twenties_boost",
            "dating_experience": "current",
            "image_selection_importance": "4",
            "honest": "yes",
        },
    )
    client.post(
        "/persona",
        data={
            f"persona::{category.key}": "no_preference"
            for category in PERSONA_CATEGORIES
        },
    )
    page = client.get("/persona/candidates")
    image_id = _first_image_id(page.text)
    client.post(
        "/persona/candidates",
        data={
            "batch_id": _hidden_value(page.text, "batch_id"),
            "selected_image_id": image_id,
            "action": "selected",
            "reaction_time_sec": "0.5",
        },
    )
    client.post(
        "/persona/confirm",
        data={"initial_rating": "7", "selection_confidence": "5"},
    )


def _start_and_show_first_round(client: TestClient) -> tuple[str, str]:
    response = client.post(
        "/experiment1/start",
        data={"block_index": "0"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    page = client.get("/experiment1/round?block=0")
    assert page.status_code == 200
    return _hidden_value(page.text, "round_id"), _first_image_id(page.text)


def _submit_first_round(client: TestClient, round_id: str, image_id: str) -> None:
    response = client.post(
        "/experiment1/round",
        data={
            "block_index": "0",
            "round_id": round_id,
            "selected_image_id": image_id,
            "preference_rating": "8",
            "difficulty": "3",
            "reaction_time_sec": "1.0",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_two_participants_are_isolated_and_resume(tmp_path):
    config = load_config(demo_override=True)
    config.database._values["url"] = f"sqlite:///{tmp_path / 'isolated.sqlite3'}"
    for name in ("data_dir", "cache_dir", "output_dir", "model_dir"):
        config.paths._values[name] = str(tmp_path / name)
    config.generator._values["output_resolution"] = 16
    config.search._values["pca_dimensions"] = 4
    config.search._values["pca_fit_samples"] = 8
    config.experiment._values["m_list"] = [2]
    config.experiment._values["rounds_per_m"] = 2
    config.experiment._values["candidate_pool_size"] = 16
    config.experiment._values["candidate_pool_multiplier"] = 2
    config.persona._values["mixture_particle_count"] = 512
    config.query._values["posterior_mc_samples"] = 32
    config.query._values["rc_posterior_samples"] = 512
    config.query._values["num_restarts"] = 1
    config.query._values["optimization_steps"] = 5
    reset_database_for_tests(config.database.url)
    app = create_app(config)

    with TestClient(app) as first, TestClient(app) as second:
        _complete_persona(first)
        _complete_persona(second)
        with get_session() as db:
            sessions = list(db.scalars(select(ExperimentSession)))
            assert len(sessions) == 2
            assert len({item.participant_id for item in sessions}) == 2

        first_round, first_image = _start_and_show_first_round(first)
        second_round, second_image = _start_and_show_first_round(second)
        _submit_first_round(first, first_round, first_image)
        _submit_first_round(second, second_round, second_image)

        resumed = first.get("/experiment1/round?block=0")
        assert resumed.status_code == 200
        assert _hidden_value(resumed.text, "round_id") == "2"

    with get_session() as db:
        assert db.scalar(select(func.count(Selection.id))) == 2
        assert db.scalar(select(func.count(ExperimentRound.round_id))) == 2
        blocks = list(db.scalars(select(ExperimentBlock)))
        assert len(blocks) == 2
        assert len({block.session_id for block in blocks}) == 2
        assert len({block.strategy_state_path for block in blocks}) == 2

        histories = []
        for block in blocks:
            with Path(block.strategy_state_path).open("rb") as handle:
                state = pickle.load(handle)
            assert len(state["history"]) == 1
            histories.append(np.asarray(state["history"][0]["queries"]))

            round_row = db.scalar(
                select(ExperimentRound).where(
                    ExperimentRound.block_id == block.block_id
                )
            )
            assert round_row.status == "completed"
            assert round_row.session_id == block.session_id
            images = list(
                db.scalars(
                    select(LatentImage).where(
                        LatentImage.image_id.in_(round_row.generated_image_ids)
                    )
                )
            )
            assert images
            assert {image.participant_id for image in images} == {
                block.participant_id
            }
            selection = db.scalar(
                select(Selection).where(Selection.block_id == block.block_id)
            )
            assert json.loads(selection.shown_image_ids) == (
                round_row.generated_image_ids
            )

        assert not np.array_equal(histories[0], histories[1])
