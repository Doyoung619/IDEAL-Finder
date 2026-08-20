from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the authenticated GPU contract with deterministic demo models."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "mock_gpu.sqlite3",
    )
    args = parser.parse_args()

    os.environ["APP_ROLE"] = "gpu"
    os.environ["APP_ENV"] = "development"
    os.environ["IDEAL_DEMO"] = "1"
    os.environ.setdefault("IDEAL_CONFIG", "configs/default.yaml")
    os.environ.setdefault(
        "GPU_GATEWAY_SECRET", "local-development-gateway-secret"
    )
    os.environ.setdefault("SESSION_SECRET", "local-development-session-secret")
    os.environ["IDEAL_DATABASE_URL"] = f"sqlite:///{args.database.resolve()}"
    os.environ.setdefault("IDEAL_ARTIFACT_STORAGE", "database")

    import uvicorn

    uvicorn.run(
        "app.entrypoint:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
