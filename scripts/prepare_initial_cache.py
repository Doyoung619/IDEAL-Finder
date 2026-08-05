"""Prepare deterministic local startup caches for the first experiment round.

The generated files stay under data/precomputed/ and are intentionally not
checked into Git. See INITIAL_RUNTIME_CACHE.md for the setup contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.generation_service import (
    generate_filtered_prior_candidates,
    save_precomputed_candidates,
)
from app.services.runtime import ExperimentRuntime
from app.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--gender",
        choices=("female", "male"),
        action="append",
        dest="genders",
        help="Prepare only one gender; repeat to prepare both.",
    )
    args = parser.parse_args()
    if args.candidate_count < 4:
        parser.error("--candidate-count must be at least 4")

    config = load_config()
    runtime = ExperimentRuntime(config)
    runtime.ensure_ready()
    genders = args.genders or ["female", "male"]
    for offset, gender in enumerate(genders):
        seed = args.seed + offset * 1009
        print(f"Preparing {gender} startup candidates (seed={seed})...", flush=True)
        candidates = generate_filtered_prior_candidates(
            runtime,
            gender,
            "east_asian_only",
            "twenties_boost",
            count=args.candidate_count,
            seed=seed,
            use_precomputed_cache=False,
        )
        path = save_precomputed_candidates(runtime, candidates, gender)
        print(f"Saved {len(candidates)} candidates to {Path(path)}", flush=True)


if __name__ == "__main__":
    main()
