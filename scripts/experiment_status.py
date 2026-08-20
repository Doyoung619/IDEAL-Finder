from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import func, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.base import get_session, init_database
from app.models import ExperimentRound, ExperimentSession
from app.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show aggregate IDEAL-Finder experiment progress."
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    init_database(config.database.url)

    with get_session() as db:
        session_counts = dict(
            db.execute(
                select(ExperimentSession.status, func.count())
                .group_by(ExperimentSession.status)
                .order_by(ExperimentSession.status)
            ).all()
        )
        round_counts = dict(
            db.execute(
                select(ExperimentRound.status, func.count())
                .group_by(ExperimentRound.status)
                .order_by(ExperimentRound.status)
            ).all()
        )
        last_session_timestamp = db.scalar(
            select(func.max(ExperimentSession.created_at))
        )

    print(f"Total sessions: {sum(session_counts.values())}")
    for status in ("completed", "in_progress", "abandoned"):
        print(f"  {status}: {session_counts.pop(status, 0)}")
    for status, count in session_counts.items():
        print(f"  {status}: {count}")
    print(f"Total rounds: {sum(round_counts.values())}")
    for status, count in round_counts.items():
        print(f"  {status}: {count}")
    print(
        "Last session timestamp: "
        f"{last_session_timestamp.isoformat() if last_session_timestamp else 'none'}"
    )


if __name__ == "__main__":
    main()
