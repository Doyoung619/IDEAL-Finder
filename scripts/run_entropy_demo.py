from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.conditional_prior import ConditionalPCAPrior
from core.entropy_query import EntropyQueryConfig, EntropyQuerySelector
from core.generator import StyleGAN2ADAGenerator
from core.mock_models import MockStyleGANGenerator
from core.preference_posterior import GaussianPreferencePosterior
from core.utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    """Create the entropy-query demo argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run M-way mutual-information query synthesis under the current "
            "distance-based softmax preference model."
        )
    )
    parser.add_argument(
        "--query-algorithm", choices=("entropy",), default="entropy"
    )
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--stylegan-pkl", type=Path)
    parser.add_argument(
        "--stylegan-repo",
        type=Path,
        default=PROJECT_ROOT / "models" / "stylegan2-ada-pytorch",
    )
    parser.add_argument("--num-options", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--posterior-mc-samples", type=int, default=512)
    parser.add_argument("--num-restarts", type=int, default=8)
    parser.add_argument("--optimization-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument(
        "--noise-mode", choices=("const", "random", "none"), default="const"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic CPU decoder without external checkpoints.",
    )
    return parser


def _generator(args: argparse.Namespace, prior: ConditionalPCAPrior):
    if args.mock:
        return MockStyleGANGenerator(prior.w_dimension)
    if args.stylegan_pkl is None:
        raise RuntimeError("--stylegan-pkl is required unless --mock is used")
    return StyleGAN2ADAGenerator(
        repo_path=str(args.stylegan_repo),
        network_path=str(args.stylegan_pkl),
        device=args.device,
        batch_size=args.batch_size,
        truncation_psi=args.truncation_psi,
        noise_mode=args.noise_mode,
    )


def _save_images(images: torch.Tensor, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    arrays = images.mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    for index, array in enumerate(arrays):
        Image.fromarray(array, "RGB").save(
            directory / f"query_{index + 1:02d}.png"
        )


def main() -> None:
    """Run a seeded synthetic preference interaction and save every round."""
    args = build_parser().parse_args()
    prior = ConditionalPCAPrior.load(args.prior)
    generator = _generator(args, prior)
    posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    selector = EntropyQuerySelector(
        EntropyQueryConfig(
            posterior_mc_samples=args.posterior_mc_samples,
            num_restarts=args.num_restarts,
            optimization_steps=args.optimization_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=resolve_device(args.device),
        )
    )
    random = np.random.default_rng(args.seed)
    true_preference = prior.sample_theta(1, generator=random)[0]
    args.output.mkdir(parents=True, exist_ok=True)
    run_log: list[dict] = []
    for round_id in range(1, args.rounds + 1):
        query_center = posterior.map_estimate.copy()
        result = selector.select(
            posterior,
            prior.theta_covariance,
            args.num_options,
            seed=args.seed + round_id,
        )
        query_points = result.query_points
        w_queries = prior.theta_to_w(query_points)
        round_directory = args.output / f"round_{round_id:02d}"
        round_directory.mkdir(parents=True, exist_ok=True)
        np.save(round_directory / "query_points.npy", query_points)
        np.save(round_directory / "query_center.npy", query_center)
        _save_images(
            generator.synthesize_w(w_queries),
            round_directory / "decoded_images",
        )
        probabilities = posterior.choice_probabilities(
            query_points, estimate=true_preference
        )
        selected = int(random.choice(args.num_options, p=probabilities))
        posterior.update(query_points, selected)
        np.save(round_directory / "posterior_mean.npy", posterior.mean)
        np.save(
            round_directory / "posterior_covariance.npy", posterior.covariance
        )
        np.save(round_directory / "map_estimate.npy", posterior.map_estimate)
        metrics = {
            "round": round_id,
            "query_center": query_center.tolist(),
            **result.metadata(),
        }
        (round_directory / "entropy_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        choice = {
            "round": round_id,
            "observed_choice": selected,
            "choice_probabilities": probabilities.tolist(),
        }
        (round_directory / "observed_choice.json").write_text(
            json.dumps(choice, indent=2), encoding="utf-8"
        )
        run_log.append({**metrics, **choice})
    (args.output / "run_log.json").write_text(
        json.dumps(run_log, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.rounds} entropy-query rounds to {args.output}")


if __name__ == "__main__":
    main()
