from __future__ import annotations

import io
import zipfile

from sqlalchemy import select

from app.db.base import get_session, reset_database_for_tests
from app.models import Participant, ScreenEvent
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
            assert "ITD-TEST0001" in archive.read("participants.csv").decode("utf-8")

