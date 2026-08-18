from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import load_config
from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.demographic_classifier import FairFaceViTDemographicClassifier
from core.generator import create_generator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a race-unrestricted gender + FairFace age 20-29 PCA prior."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/persona_study.yaml")
    parser.add_argument("--gender", choices=("female", "male"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted", type=int, default=2500)
    parser.add_argument("--max-generated", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gender-threshold", type=float, default=0.80)
    parser.add_argument("--age-threshold", type=float, default=0.55)
    parser.add_argument("--dimension", type=int, default=12)
    parser.add_argument("--coordinate-clip", type=float, default=1.5)
    parser.add_argument("--sampling-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config, demo_override=False)
    generator = create_generator(config)
    classifier = FairFaceViTDemographicClassifier(
        config.demographic.race_weights,
        config.demographic.gender_weights,
        config.demographic.age_weights,
        devices=list(config.demographic.devices),
        batch_size=args.batch_size,
    )
    gender_index = 1 if args.gender == "female" else 0
    random = np.random.default_rng(args.seed)
    selected: list[np.ndarray] = []
    generated = 0
    started = time.monotonic()

    while sum(len(batch) for batch in selected) < args.accepted and generated < args.max_generated:
        count = min(args.batch_size, args.max_generated - generated)
        z = random.standard_normal((count, generator.z_dim)).astype(np.float32)
        w = generator.map_z_to_w(z)
        images = generator.decode(w)
        probabilities = classifier.predict_proba(images)
        gender_probability = probabilities["gender"][:, gender_index].numpy()
        age_20_29_probability = probabilities["age"][:, 3].numpy()
        mask = (
            (gender_probability >= args.gender_threshold)
            & (age_20_29_probability >= args.age_threshold)
        )
        if np.any(mask):
            selected.append(np.asarray(w[mask], dtype=np.float32))
        generated += count
        accepted = sum(len(batch) for batch in selected)
        if generated % (args.batch_size * 10) == 0 or accepted >= args.accepted:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"{args.gender}: generated={generated}, accepted={accepted}/{args.accepted}, "
                f"rate={accepted / generated:.3f}, speed={generated / elapsed:.1f}/s",
                flush=True,
            )

    values = np.concatenate(selected, axis=0)[: args.accepted]
    if len(values) < args.accepted:
        raise RuntimeError(
            f"Only {len(values)} gender + age 20-29 samples passed after {generated} generations"
        )
    pca = PCA(
        n_components=args.dimension,
        svd_solver="randomized",
        random_state=args.seed,
    ).fit(values)
    prior = ConditionalPCAPrior(
        condition=DemographicCondition(args.gender, ()),
        mu_w=pca.mean_,
        components=pca.components_,
        eigenvalues=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        accepted_samples=len(values),
        thresholds={
            "gender": args.gender_threshold,
            "age_20_29": args.age_threshold,
        },
        generator_metadata={
            "name": generator.generator_name,
            "checkpoint": str(config.generator.network_path),
            "frozen": True,
            "target_definition": f"{args.gender}, any race, FairFace age 20-29",
            "generated_samples": generated,
        },
        seed=args.seed,
        coordinate_clip=args.coordinate_clip,
        sampling_scale=args.sampling_scale,
    )
    prior.save(args.output)
    print(
        f"Saved {args.dimension}D {args.gender} + age 20-29, race-unrestricted prior "
        f"to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
