from __future__ import annotations

import io
import csv
import json
import zipfile

from sqlalchemy import select

from app.db.base import get_session, reset_database_for_tests
from app.models import (
    ExperimentBlock,
    FaceRating,
    LatentImage,
    Participant,
    ScreenEvent,
    Selection,
)
from app.services.export_service import build_csv_export_zip
from experiments.logging_utils import enter_screen, exit_screen


def test_screen_logging_and_csv_export(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'test.sqlite3'}"
    reset_database_for_tests(database_url)
    with get_session() as db:
        participant = Participant(
            participant_id="ITD-TEST0001",
            status="created",
            m_condition_order="[2,4,8,16]",
            recommendation_condition_order='["none","basic_info","click_profile","selection_history"]',
            base_seed=1,
        )
        db.add(participant)
        db.flush()
        block = ExperimentBlock(
            block_id="ITD-TEST0001-B1-M2",
            participant_id=participant.participant_id,
            sequence_index=0,
            m_value=2,
            strategy_mode="rc_mlq",
            initial_state_id="state-1",
            initial_seed=7,
            mu_path="mu.npy",
            sigma=1.0,
            strategy_state_path="state.pkl",
            strategy_parameters_json="{}",
        )
        db.add(block)
        db.flush()
        image = LatentImage(
            image_id="00000000-0000-0000-0000-000000000001",
            participant_id=participant.participant_id,
            block_id=block.block_id,
            round_id=1,
            stage_type="experiment1",
            latent_path="query.npy",
            image_path="query.png",
            generator_type="demo",
            generator_seed=7,
            score_metadata_json=json.dumps(
                {
                    "display_index": 1,
                    "search_version": "test-v1",
                    "proposal_backend": "rc_mlq",
                    "query_metrics": {
                        "beta": 1.2,
                        "min_pairwise_theta_distance": 0.4,
                        "diversity_guard_triggered": True,
                    },
                }
            ),
        )
        db.add(image)
        db.flush()
        db.add(
            Selection(
                participant_id=participant.participant_id,
                block_id=block.block_id,
                round_id=1,
                m_value=2,
                shown_image_ids=json.dumps([image.image_id]),
                selected_image_id=image.image_id,
                reaction_time_sec=1.5,
                difficulty_rating=3,
            )
        )
        db.add(
            FaceRating(
                participant_id=participant.participant_id,
                stage_type="experiment1_selection",
                image_id=image.image_id,
                rating_1_10=8,
                batch_id=f"{block.block_id}-round-001",
                condition_type="M=2",
            )
        )
        db.commit()

        enter_screen(db, participant.participant_id, "consent")
        exit_screen(db, participant.participant_id, "consent")
        event = db.scalar(select(ScreenEvent))
        assert event is not None
        assert event.exit_time is not None

        payload = build_csv_export_zip(db)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            assert "participants.csv" in archive.namelist()
            assert "screens.csv" in archive.namelist()
            assert "persona_profiles.csv" in archive.namelist()
            assert "persona_candidate_batches.csv" in archive.namelist()
            assert "persona_initializations.csv" in archive.namelist()
            assert "final_refinement_evaluations.csv" in archive.namelist()
            assert "analysis_rounds.csv" in archive.namelist()
            assert "rounds_flat.csv" in archive.namelist()
            assert "experiment_sessions.csv" in archive.namelist()
            assert "experiment_rounds.csv" in archive.namelist()
            assert "experiment_events.csv" in archive.namelist()
            assert "README.txt" in archive.namelist()
            assert "ITD-TEST0001" in archive.read("participants.csv").decode("utf-8")
            analysis_rows = list(
                csv.DictReader(
                    io.StringIO(
                        archive.read("analysis_rounds.csv").decode("utf-8")
                    )
                )
            )
            assert len(analysis_rows) == 1
            assert analysis_rows[0]["algorithm"] == "rc_mlq"
            assert analysis_rows[0]["preference_rating_1_10"] == "8"
            assert analysis_rows[0]["diversity_guard_triggered"] == "True"
