from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.base import init_database
from app.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the current IDEAL-Finder schema in an empty database."
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    engine = init_database(config.database.url)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    print("Database schema is ready.")


if __name__ == "__main__":
    main()
