from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.generation_service import (
    evaluate_candidates,
    generate_filtered_prior_candidates,
    strict_candidate_indices,
)
from app.services.runtime import ExperimentRuntime
from app.settings import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated real-StyleGAN MVLQ updates with hard filters."
    )
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--target-gender",
        choices=("female", "male", "any"),
        default="female",
    )
    parser.add_argument(
        "--age-appearance",
        choices=("twenties_boost", "twenties_thirties", "any_adult"),
        default="twenties_boost",
    )
    args = parser.parse_args()

    config = load_config()
    if config.generator.mode != "stylegan2_ada":
        raise RuntimeError("Unset IDEAL_DEMO before running this production test.")
    rounds = args.rounds or config.experiment.rounds_per_m
    target_gender = None if args.target_gender == "any" else args.target_gender
    runtime = ExperimentRuntime(config)
    runtime.ensure_ready()
    destination = Path(config.paths.output_dir) / "stylegan_stress_mvlq"
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "mvlq_state.pkl"
    state_path.unlink(missing_ok=True)
    center = generate_filtered_prior_candidates(
        runtime,
        target_gender=target_gender,
        target_face_region="east_asian_only",
        target_age_appearance=args.age_appearance,
        count=1,
        seed=config.experiment.seed + 9000,
    )[0].latent
    strategy = runtime.strategy_for("mvlq", {})

    total_shown = 0
    for round_id in range(1, rounds + 1):
        seed = config.experiment.seed + 9000 + round_id
        for attempt in range(12):
            query_scale = max(0.01, 0.70**attempt)
            proposal = strategy.propose(
                center=center,
                sigma=query_scale,
                count=args.m,
                seed=seed,
                state_path=str(state_path),
                round_id=round_id,
                display_count=args.m,
            )
            evaluated = evaluate_candidates(runtime, proposal.latents)
            valid_indices = strict_candidate_indices(
                runtime,
                evaluated,
                target_gender=target_gender,
            )
            if len(valid_indices) == args.m:
                break
        else:
            raise RuntimeError(
                f"Round {round_id}: MVLQ could not produce {args.m} hard-valid "
                "East Asian portraits."
            )

        round_dir = destination / f"round_{round_id:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for display_index, candidate in enumerate(evaluated, start=1):
            candidate.image.save(round_dir / f"face_{display_index:02d}.png")
        winner_index = len(proposal.latents) - 1
        center, _ = strategy.update(
            center=center,
            sigma=query_scale,
            shown_latents=proposal.latents,
            winner_latent=proposal.latents[winner_index],
            state_path=str(state_path),
        )
        center_evaluation = evaluate_candidates(runtime, center[None, :])
        if not strict_candidate_indices(
            runtime,
            center_evaluation,
            target_gender=target_gender,
        ):
            center = proposal.latents[winner_index].copy()
            strategy.constrain_map(center, str(state_path))
        total_shown += len(proposal.latents)
        print(
            f"round={round_id:02d} east_asian_faces={len(valid_indices):02d}/"
            f"{len(evaluated):02d} shown={len(proposal.latents):02d} "
            f"query_scale={query_scale:.4f}"
        )

    print(
        f"PASS: {total_shown} MVLQ stimuli across {rounds} rounds passed the "
        f"hard East Asian, face, adult, portrait, and gender filters. "
        f"Output: {destination}"
    )


if __name__ == "__main__":
    main()
