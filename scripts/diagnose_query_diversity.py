from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import load_config
from core.clip_ranker import CLIPRanker
from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.generator import create_generator
from core.preference_posterior import ParticleMixturePreferencePosterior
from core.query_diversity import QueryDiversityConfig, ensure_query_diversity
from core.query_strategy import (
    EntropyQueryConfig,
    EntropyQuerySelector,
    _NumpyEntropyQuerySelector,
    _SelectorConfig,
)
from core.rc_mlq import RCMLQConfig, RCMLQSelector


UNCERTAINTY = {"large": 1.0, "medium": 0.35, "small": 0.08, "very_small": 0.015}


def contact_sheet(images: list[Image.Image], title: str, path: Path) -> None:
    width, height = images[0].size
    canvas = Image.new("RGB", (4 * width, 2 * height + 32), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 4) * width, 32 + (index // 4) * height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Entropy/RC-MLQ diversity diagnostics across shrinking posteriors."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/persona_study.yaml")
    parser.add_argument("--gender", choices=("female", "male"), default="female")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/query_diversity_diagnostic")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="Optional trial threshold; omit for diagnostic-only baseline output.",
    )
    args = parser.parse_args()

    config = load_config(args.config, demo_override=args.mock)
    generator = create_generator(config)
    embedder = CLIPRanker(
        enabled=bool(config.clip.enabled),
        model_name=str(config.clip.model_name),
        pretrained=str(config.clip.pretrained),
        device=str(config.clip.device),
        batch_size=int(config.clip.batch_size),
        require_real=not args.mock,
        allow_mock=args.mock,
    )
    races = list(config.demographic.race_targets)
    if args.mock:
        dimension = min(int(config.conditional_prior.dimension), generator.latent_dim)
        prior = ConditionalPCAPrior(
            condition=DemographicCondition(args.gender, tuple(races)),
            mu_w=np.zeros(generator.latent_dim),
            components=np.eye(generator.latent_dim)[:dimension],
            eigenvalues=np.ones(dimension),
            explained_variance_ratio=np.full(dimension, 1.0 / dimension),
            accepted_samples=1,
            coordinate_clip=1.5,
        )
    else:
        prior_path = str(config.conditional_prior.artifact_path).format(
            gender=args.gender, race="_".join(races)
        )
        prior = ConditionalPCAPrior.load(prior_path)
    query = config.query
    entropy_config = dict(
        posterior_mc_samples=min(int(query.posterior_mc_samples), 256),
        num_restarts=min(int(query.num_restarts), 3),
        optimization_steps=min(int(query.optimization_steps), 60),
        learning_rate=float(query.learning_rate),
        seed=int(query.seed),
        device=str(config.generator.device),
        output_spread_scale=float(query.entropy_output_spread_scale),
    )
    selector_config = EntropyQueryConfig or _SelectorConfig
    selector_class = EntropyQuerySelector or _NumpyEntropyQuerySelector
    entropy = selector_class(selector_config(**entropy_config))
    rc_mlq = RCMLQSelector(
        RCMLQConfig(
            resolution_min=float(query.rc_resolution_min),
            resolution_max=float(query.rc_resolution_max),
            resolution_steps=int(query.rc_resolution_steps),
            posterior_samples=max(512, min(int(query.rc_posterior_samples), 1024)),
            beta=float(query.beta),
            coordinate_clip=prior.coordinate_clip,
        )
    )
    guard_values = config.query_diversity.as_dict()
    if args.similarity_threshold is not None:
        guard_values["image_similarity_threshold"] = args.similarity_threshold
    records = []
    for stage_index, (stage, scale) in enumerate(UNCERTAINTY.items()):
        covariance = np.asarray(prior.theta_covariance) * scale
        posterior = ParticleMixturePreferencePosterior.from_gaussian_mixture(
            local_mean=np.asarray(prior.theta_mean),
            local_covariance=covariance,
            global_mean=np.asarray(prior.theta_mean),
            global_covariance=covariance,
            global_weight=0.45,
            particle_count=1024,
            seed=3100 + stage_index,
            beta=float(query.beta),
        )
        proposals = {
            "entropy": entropy.select(
                posterior, prior.theta_covariance, 8, seed=4100 + stage_index
            ).query_points,
            "rc_mlq": rc_mlq.select(posterior, 8, seed=5100 + stage_index).query_points,
        }
        for algorithm, raw in proposals.items():
            for enabled in (False, True):
                final, metrics = ensure_query_diversity(
                    raw,
                    posterior.covariance,
                    prior,
                    generator,
                    embedder,
                    QueryDiversityConfig(**{**guard_values, "enabled": enabled}),
                    posterior_particles=posterior.particles,
                    posterior_weights=posterior.weights,
                    beta=posterior.beta,
                )
                images = generator.decode(prior.theta_to_w(final))
                label = "guard" if enabled else "baseline"
                contact_sheet(
                    images,
                    f"{algorithm} · {stage} · {label}",
                    args.output / f"{algorithm}_{stage}_{label}.png",
                )
                records.append(
                    {"algorithm": algorithm, "uncertainty": stage, "scale": scale, "mode": label, **metrics}
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for row, algorithm in enumerate(("entropy", "rc_mlq")):
        for column, stage in enumerate(UNCERTAINTY):
            record = next(
                item for item in records
                if item["algorithm"] == algorithm
                and item["uncertainty"] == stage
                and item["mode"] == "baseline"
            )
            axes[row, column].hist(
                record["pairwise_image_similarities"],
                bins=np.linspace(-1.0, 1.0, 41),
            )
            axes[row, column].set_title(f"{algorithm} · {stage}")
    figure.supxlabel(
        "Mock embedding cosine similarity"
        if args.mock
        else "OpenCLIP cosine similarity"
    )
    figure.supylabel("Pair count")
    figure.tight_layout()
    figure.savefig(args.output / "pairwise_similarity_histograms.png", dpi=160)
    plt.close(figure)
    print(f"Saved {len(records)} diagnostic rows and contact sheets to {args.output}")


if __name__ == "__main__":
    main()
