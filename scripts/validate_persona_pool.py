from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.persona_pool import PersonaPool


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a persona pool artifact.")
    parser.add_argument("pool", type=Path)
    parser.add_argument("--minimum-size", type=int, default=1000)
    args = parser.parse_args()
    pool = PersonaPool.load(args.pool, minimum_size=args.minimum_size)
    print(
        f"Valid pool: size={pool.size}, theta_dim={pool.theta_dimension}, "
        f"embedding_dim={pool.clip_image_embedding.shape[1]}, "
        f"version={pool.metadata.get('pool_version', 'unknown')}"
    )


if __name__ == "__main__":
    main()
