from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.base import get_session, init_database
from app.services.export_service import write_csv_exports
from app.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all IDEAL-Finder experiment logs as a CSV ZIP archive."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    init_database(config.database.url)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output or (
        Path(config.paths.output_dir) / "exports" / f"ideal_finder_{timestamp}.zip"
    )
    with get_session() as db:
        path = write_csv_exports(db, destination)
    print(f"Exported experiment logs to {path}")


if __name__ == "__main__":
    main()
