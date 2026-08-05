from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.conditional_prior import DemographicCondition
from core.demographic_classifier import FairFaceDemographicClassifier
from core.demographic_prior_builder import DemographicPriorBuilder, PriorBuildConfig
from core.generator import StyleGAN2ADAGenerator
from core.mock_models import MockDemographicClassifier, MockStyleGANGenerator


def build_parser() -> argparse.ArgumentParser:
    """Create the conditional-prior builder argument parser."""
    parser = argparse.ArgumentParser(
        description="Build a demographic-conditioned StyleGAN W-space PCA prior."
    )
    parser.add_argument("--stylegan-pkl", type=Path)
    parser.add_argument(
        "--stylegan-repo",
        type=Path,
        default=PROJECT_ROOT / "models" / "stylegan2-ada-pytorch",
    )
    parser.add_argument("--fairface-weights", type=Path)
    parser.add_argument("--gender", choices=("female", "male"), required=True)
    parser.add_argument(
        "--race",
        action="append",
        choices=(
            "east_asian",
            "southeast_asian",
            "white",
            "black",
            "indian",
            "middle_eastern",
            "latino_hispanic",
        ),
        dest="races",
    )
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--selection-mode", choices=("hard", "soft"), default="hard")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--min-accepted", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gender-threshold", type=float, default=0.90)
    parser.add_argument("--race-threshold", type=float, default=0.80)
    parser.add_argument("--covariance-eps", type=float, default=1e-6)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--noise-mode", choices=("const", "random", "none"), default="const")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-grid", type=Path)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic CPU mock models; no checkpoints are required.",
    )
    return parser


def _models(args: argparse.Namespace):
    if args.mock:
        w_dimension = max(args.latent_dim + 2, 8)
        return MockStyleGANGenerator(w_dimension), MockDemographicClassifier()
    if args.stylegan_pkl is None:
        raise RuntimeError("--stylegan-pkl is required unless --mock is used")
    if args.fairface_weights is None:
        raise RuntimeError("--fairface-weights is required unless --mock is used")
    generator = StyleGAN2ADAGenerator(
        repo_path=str(args.stylegan_repo),
        network_path=str(args.stylegan_pkl),
        device=args.device,
        batch_size=args.batch_size,
        truncation_psi=args.truncation_psi,
        noise_mode=args.noise_mode,
    )
    classifier = FairFaceDemographicClassifier(
        args.fairface_weights, device=args.device, batch_size=args.batch_size
    )
    return generator, classifier


def _save_grid(images: torch.Tensor, destination: Path) -> None:
    count = min(len(images), 16)
    columns = min(4, count)
    rows = int(np.ceil(count / columns))
    height, width = images.shape[2:]
    canvas = Image.new("RGB", (columns * width, rows * height))
    arrays = images[:count].mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    for index, array in enumerate(arrays):
        canvas.paste(Image.fromarray(array, "RGB"), ((index % columns) * width, (index // columns) * height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> None:
    """Build and save one conditional prior artifact."""
    args = build_parser().parse_args()
    generator, classifier = _models(args)
    condition = DemographicCondition(
        gender=args.gender,
        race_targets=tuple(args.races or ["east_asian"]),
    )
    config = PriorBuildConfig(
        condition=condition,
        latent_dimension=args.latent_dim,
        selection_mode=args.selection_mode,
        num_generator_samples=args.num_samples,
        min_accepted_samples=args.min_accepted,
        batch_size=args.batch_size,
        gender_threshold=args.gender_threshold,
        race_threshold=args.race_threshold,
        covariance_eps=args.covariance_eps,
        covariance_shrinkage=args.covariance_shrinkage,
        seed=args.seed,
        cache_path=str(args.cache) if args.cache else None,
    )
    prior = DemographicPriorBuilder(generator, classifier).build(config)
    prior.save(args.output)
    if args.debug_grid:
        random = np.random.default_rng(args.seed + 1)
        theta = prior.sample_theta(16, generator=random)
        _save_grid(generator.generate_from_theta(theta, prior), args.debug_grid)
    print(
        f"Saved {prior.dimension}D prior with {prior.accepted_samples} samples "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
