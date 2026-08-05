from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import get_session, init_database
from app.services.export_service import write_csv_exports
from app.settings import load_config


def main() -> None:
    config = load_config()
    init_database(config.database.url)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(config.paths.output_dir) / f"csv_export_{timestamp}.zip"
    with get_session() as db:
        write_csv_exports(db, destination)
    print(destination)


if __name__ == "__main__":
    main()
