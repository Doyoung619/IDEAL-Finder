from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.conditional_prior import ConditionalPCAPrior
from core.constrained_mvlq import DemographicConstrainedMVLQ
from core.demographic_classifier import FairFaceDemographicClassifier
from core.generator import StyleGAN2ADAGenerator
from core.mock_models import MockDemographicClassifier, MockStyleGANGenerator
from core.preference_posterior import GaussianPreferencePosterior


def build_parser() -> argparse.ArgumentParser:
    """Create the constrained-MVLQ demo argument parser."""
    parser = argparse.ArgumentParser(
        description="Run a simulated multi-round demographic-constrained MVLQ demo."
    )
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--stylegan-pkl", type=Path)
    parser.add_argument(
        "--stylegan-repo",
        type=Path,
        default=PROJECT_ROOT / "models" / "stylegan2-ada-pytorch",
    )
    parser.add_argument("--fairface-weights", type=Path)
    parser.add_argument("--num-options", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--distance-space", choices=("theta", "w"), default="theta")
    parser.add_argument("--gender-threshold", type=float)
    parser.add_argument("--race-threshold", type=float)
    parser.add_argument("--backtrack-factor", type=float, default=0.8)
    parser.add_argument("--min-alpha", type=float, default=0.1)
    parser.add_argument("--max-backtracking-steps", type=int, default=15)
    parser.add_argument("--center-candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--noise-mode", choices=("const", "random", "none"), default="const")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic CPU mock models; the prior must have a compatible W dimension.",
    )
    return parser


def _models(args: argparse.Namespace, prior: ConditionalPCAPrior):
    if args.mock:
        return (
            MockStyleGANGenerator(prior.w_dimension),
            MockDemographicClassifier(),
        )
    if args.stylegan_pkl is None:
        raise RuntimeError("--stylegan-pkl is required unless --mock is used")
    if args.fairface_weights is None:
        raise RuntimeError("--fairface-weights is required unless --mock is used")
    return (
        StyleGAN2ADAGenerator(
            repo_path=str(args.stylegan_repo),
            network_path=str(args.stylegan_pkl),
            device=args.device,
            batch_size=args.batch_size,
            truncation_psi=args.truncation_psi,
            noise_mode=args.noise_mode,
        ),
        FairFaceDemographicClassifier(
            args.fairface_weights,
            device=args.device,
            batch_size=args.batch_size,
        ),
    )


def _save_grid(
    images: torch.Tensor,
    selected: int,
    destination: Path,
) -> None:
    arrays = images.mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    height, width = arrays.shape[1:3]
    canvas = Image.new("RGB", (len(arrays) * width, height))
    for index, array in enumerate(arrays):
        image = Image.fromarray(array, "RGB")
        if index == selected:
            draw = ImageDraw.Draw(image)
            draw.rectangle((1, 1, width - 2, height - 2), outline="red", width=max(1, width // 64))
        canvas.paste(image, (index * width, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> None:
    """Run a seeded simulated user and persist round-level diagnostics."""
    args = build_parser().parse_args()
    if args.rounds < 1 or args.num_options < 2:
        raise ValueError("--rounds must be positive and --num-options at least two")
    prior = ConditionalPCAPrior.load(args.prior)
    generator, classifier = _models(args, prior)
    gender_threshold = args.gender_threshold or float(
        prior.thresholds.get("gender", 0.90)
    )
    race_threshold = args.race_threshold or float(
        prior.thresholds.get("race", 0.80)
    )
    posterior = GaussianPreferencePosterior.initialize_from_prior(
        prior, beta=args.beta, distance_space=args.distance_space
    )
    strategy = DemographicConstrainedMVLQ(
        prior,
        generator,
        classifier,
        gender_threshold=gender_threshold,
        race_threshold=race_threshold,
        backtrack_factor=args.backtrack_factor,
        min_alpha=args.min_alpha,
        max_backtracking_steps=args.max_backtracking_steps,
        center_candidates=args.center_candidates,
        seed=args.seed,
    )
    random = np.random.default_rng(args.seed)
    true_theta = prior.sample_theta(1, generator=random)[0].astype(np.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    run_log = {
        "condition": {
            "gender": prior.condition.gender,
            "race_targets": list(prior.condition.race_targets),
        },
        "seed": args.seed,
        "true_theta": true_theta.tolist(),
        "rounds": [],
    }
    for round_id in range(1, args.rounds + 1):
        result = strategy.propose(
            posterior_map=posterior.map_estimate,
            posterior_mean=posterior.mean,
            posterior_covariance=posterior.covariance,
            count=args.num_options,
            radius=args.radius,
        )
        choice_probabilities = posterior.choice_probabilities(
            result.theta_queries, estimate=true_theta
        )
        selected = int(random.choice(args.num_options, p=choice_probabilities))
        images = generator.synthesize_w(result.w_queries)
        _save_grid(images, selected, args.output / f"round_{round_id:02d}.png")
        posterior.update(result.theta_queries, selected)
        record = {
            "round": round_id,
            **result.metadata(),
            "selected_choice": selected,
            "choice_probabilities": choice_probabilities.tolist(),
            "posterior_map": posterior.map_estimate.tolist(),
            "posterior_covariance_eigenvalues": np.linalg.eigvalsh(
                posterior.covariance
            ).tolist(),
            "estimation_error": float(
                np.linalg.norm(posterior.map_estimate - true_theta)
            ),
        }
        run_log["rounds"].append(record)
        (args.output / f"round_{round_id:02d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
    (args.output / "run_log.json").write_text(
        json.dumps(run_log, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.rounds} rounds to {args.output}")


if __name__ == "__main__":
    main()
