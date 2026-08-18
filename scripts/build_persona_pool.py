from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import load_config
from core.clip_ranker import CLIPRanker
from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.demographic_classifier import (
    FairFaceDemographicClassifier,
    FairFaceViTDemographicClassifier,
)
from core.face_filter import FaceQualityFilter
from core.generator import DemoFaceGenerator, create_generator
from core.persona_pool import PersonaPool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a resumable hard-filtered persona warm-start pool."
    )
    parser.add_argument("--gender", choices=("female", "male"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--target-size", type=int)
    parser.add_argument("--max-generated", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", help="Use deterministic demo models")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _config_hash(values: dict) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, generated: int, accepted: dict[str, list]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            generated=np.asarray([generated], dtype=np.int64),
            theta=np.asarray(accepted["theta"], dtype=np.float32),
            w=np.asarray(accepted["w"], dtype=np.float32),
            clip_image_embedding=np.asarray(accepted["clip"], dtype=np.float32),
            quality_score=np.asarray(accepted["quality"], dtype=np.float32),
            gender_probability=np.asarray(accepted["gender"], dtype=np.float32),
            age_20_29_probability=np.asarray(accepted["age"], dtype=np.float32),
            generator_seed=np.asarray(accepted["seed"], dtype=np.int64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> tuple[int, dict[str, list]]:
    with np.load(path, allow_pickle=False) as archive:
        accepted = {
            "theta": list(archive["theta"]),
            "w": list(archive["w"]),
            "clip": list(archive["clip_image_embedding"]),
            "quality": list(archive["quality_score"]),
            "gender": list(archive["gender_probability"]),
            "age": list(archive["age_20_29_probability"]),
            "seed": list(archive["generator_seed"]),
        }
        return int(archive["generated"][0]), accepted


def _render_progress(path: Path, state: dict) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), facecolor="#0b1220")
    axis.set_facecolor("#0b1220")
    axis.axis("off")
    generated = state["generated"]
    accepted = state["accepted"]
    target = state["target"]
    rate = accepted / generated if generated else 0.0
    elapsed = state["elapsed_seconds"]
    speed = generated / elapsed if elapsed > 0 else 0.0
    if accepted > 0 and speed > 0 and rate > 0:
        eta = max(0.0, target - accepted) / (speed * rate)
        eta_text = f"{eta / 60:.1f} min"
    else:
        eta_text = "estimating..."
    lines = [
        ("Target", f"{state['gender']} · any race · age 20–29"),
        ("Generated", f"{generated:,}"),
        ("Accepted", f"{accepted:,} / {target:,}"),
        ("Acceptance", f"{100 * rate:.2f}%"),
        ("Throughput", f"{speed:.2f} samples/sec"),
        ("Elapsed", f"{elapsed / 60:.1f} min"),
        ("ETA", eta_text),
        ("Gender pass", f"{100 * state['gender_pass'] / max(generated, 1):.2f}%"),
        ("Age 20–29 pass", f"{100 * state['age_pass'] / max(generated, 1):.2f}%"),
        ("Face-quality pass", f"{100 * state['quality_pass'] / max(generated, 1):.2f}%"),
    ]
    axis.text(0.05, 0.93, "Persona Pool Build Progress", color="white", fontsize=21, weight="bold")
    for index, (label, value) in enumerate(lines):
        y = 0.84 - index * 0.07
        axis.text(0.06, y, label, color="#8fa7c2", fontsize=11)
        axis.text(0.43, y, value, color="#e8eef7", fontsize=12, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, demo_override=args.mock)
    target = int(args.target_size or config.persona_pool.target_size)
    max_generated = int(args.max_generated or config.persona_pool.max_generated)
    batch_size = int(args.batch_size or config.persona_pool.batch_size)
    if target < 1 or max_generated < target or batch_size < 1:
        raise ValueError("target, max-generated, and batch-size must be positive and consistent")
    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "gender": args.gender,
        "target": target,
        "max_generated": max_generated,
        "batch_size": batch_size,
        "seed": args.seed,
        "mock": args.mock,
        "gender_threshold": float(config.persona_pool.gender_threshold),
        "age_threshold": float(config.persona_pool.age_20_29_threshold),
        "minimum_quality": float(config.persona_pool.minimum_quality),
        "exactly_one_face": bool(config.persona_pool.require_exactly_one_face),
        "generator_mode": str(config.generator.mode),
        "generator_checkpoint": str(config.generator.network_path),
        "prior_path": str(config.conditional_prior.artifact_path).format(
            gender=args.gender, race="unrestricted"
        ),
        "clip_model": str(config.clip.model_name),
        "clip_pretrained": str(config.clip.pretrained),
        "demographic_classifier": str(config.demographic.classifier),
        "face_detector_path": config.persona_pool.as_dict().get(
            "face_detector_path"
        ),
    }
    settings_hash = _config_hash(settings)
    metadata_path = output / "build_config.json"
    checkpoint_path = output / "pool_checkpoint.npz"
    if metadata_path.exists() and not args.no_resume:
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != settings_hash:
            raise RuntimeError("Existing persona pool checkpoint has a different config hash")
    _atomic_json(metadata_path, {"config_hash": settings_hash, **settings})

    prior_path = str(config.conditional_prior.artifact_path).format(
        gender=args.gender, race="unrestricted"
    )
    if args.mock:
        generator = DemoFaceGenerator(
            latent_dim=int(config.generator.demo_latent_dim),
            output_resolution=int(config.generator.output_resolution),
            device="cpu",
        )
        dimension = min(int(config.conditional_prior.dimension), generator.latent_dim)
        prior = ConditionalPCAPrior(
            condition=DemographicCondition(args.gender, ()),
            mu_w=np.zeros(generator.latent_dim),
            components=np.eye(generator.latent_dim)[:dimension],
            eigenvalues=np.ones(dimension),
            explained_variance_ratio=np.full(dimension, 1 / dimension),
            accepted_samples=1,
            seed=args.seed,
        )
        classifier = None
    else:
        generator = create_generator(config)
        prior = ConditionalPCAPrior.load(prior_path)
        if str(config.demographic.classifier) == "fairface_vit_triplet":
            classifier = FairFaceViTDemographicClassifier(
                config.demographic.race_weights,
                config.demographic.gender_weights,
                config.demographic.age_weights,
                devices=list(config.demographic.devices),
                batch_size=batch_size,
            )
        else:
            classifier = FairFaceDemographicClassifier(
                config.demographic.weights,
                device=config.demographic.device,
                batch_size=batch_size,
            )
    clip = CLIPRanker(
        enabled=not args.mock,
        model_name=config.clip.model_name,
        pretrained=config.clip.pretrained,
        device=config.clip.device,
        batch_size=batch_size,
        require_real=not args.mock,
        allow_mock=args.mock,
    )
    persona_pool_values = config.persona_pool.as_dict()
    quality_filter = FaceQualityFilter(
        settings["minimum_quality"],
        yunet_path=persona_pool_values.get("face_detector_path"),
        face_confidence_threshold=float(
            persona_pool_values.get("face_confidence_threshold", 0.65)
        ),
    )
    accepted = {key: [] for key in ("theta", "w", "clip", "quality", "gender", "age", "seed")}
    generated = 0
    elapsed_before = 0.0
    counters = {"gender_pass": 0, "age_pass": 0, "quality_pass": 0}
    if checkpoint_path.exists() and not args.no_resume:
        generated, accepted = _load_checkpoint(checkpoint_path)
        progress_path = output / "progress.json"
        if progress_path.exists():
            previous_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if previous_progress.get("config_hash") == settings_hash:
                for key in counters:
                    counters[key] = int(previous_progress.get(key, 0))
                elapsed_before = float(previous_progress.get("elapsed_seconds", 0.0))
    started = time.monotonic()

    def checkpoint() -> None:
        _atomic_checkpoint(checkpoint_path, generated, accepted)
        state = {
            "gender": args.gender,
            "generated": generated,
            "accepted": len(accepted["theta"]),
            "target": target,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "config_hash": settings_hash,
            **counters,
        }
        _atomic_json(output / "progress.json", state)
        _render_progress(output / "progress.png", state)

    random = np.random.default_rng(args.seed)
    if generated:
        random.standard_normal((generated, prior.dimension))
    try:
        while len(accepted["theta"]) < target and generated < max_generated:
            count = min(batch_size, max_generated - generated)
            theta = random.standard_normal((count, prior.dimension)).astype(np.float32)
            w = np.asarray(prior.theta_to_w(theta), dtype=np.float32)
            images = generator.decode(w)
            if classifier is None:
                gender_prob = age_prob = np.full(count, 0.99, dtype=np.float32)
            else:
                probabilities = classifier.predict_proba(images)
                gender_index = 1 if args.gender == "female" else 0
                gender_prob = probabilities["gender"][:, gender_index].numpy()
                age_prob = probabilities["age"][:, 3].numpy()
            qualities = [
                quality_filter.evaluate(image, require_face_detection=not args.mock)
                for image in images
            ]
            embeddings = clip.encode_images(images)
            for local in range(count):
                gender_ok = gender_prob[local] >= settings["gender_threshold"]
                age_ok = age_prob[local] >= settings["age_threshold"]
                quality_ok = qualities[local].accepted and (
                    not settings["exactly_one_face"] or qualities[local].face_count == 1
                )
                counters["gender_pass"] += int(gender_ok)
                counters["age_pass"] += int(age_ok)
                counters["quality_pass"] += int(quality_ok)
                if not (gender_ok and age_ok and quality_ok):
                    continue
                index = len(accepted["theta"])
                image_path = images_dir / f"{index:06d}.png"
                images[local].save(image_path, format="PNG", optimize=True)
                accepted["theta"].append(theta[local])
                accepted["w"].append(w[local])
                accepted["clip"].append(embeddings[local])
                accepted["quality"].append(qualities[local].score)
                accepted["gender"].append(gender_prob[local])
                accepted["age"].append(age_prob[local])
                accepted["seed"].append(args.seed + generated + local)
                if len(accepted["theta"]) >= target:
                    break
            generated += count
            checkpoint()
    except KeyboardInterrupt:
        checkpoint()
        raise SystemExit("Interrupted after saving a resumable checkpoint")

    checkpoint()
    if len(accepted["theta"]) < target:
        raise RuntimeError(
            f"Generated budget exhausted: accepted {len(accepted['theta'])}/{target}"
        )
    size = len(accepted["theta"])
    version = f"{args.gender}-{settings_hash[:12]}"
    pool = PersonaPool(
        root=output,
        pool_ids=np.asarray([f"{args.gender}-v1-{index:06d}" for index in range(size)]),
        theta=np.asarray(accepted["theta"], dtype=np.float32),
        w=np.asarray(accepted["w"], dtype=np.float32),
        clip_image_embedding=np.asarray(accepted["clip"], dtype=np.float32),
        quality_score=np.asarray(accepted["quality"], dtype=np.float32),
        gender_probability=np.asarray(accepted["gender"], dtype=np.float32),
        age_20_29_probability=np.asarray(accepted["age"], dtype=np.float32),
        generator_seed=np.asarray(accepted["seed"], dtype=np.int64),
        image_paths=tuple(f"images/{index:06d}.png" for index in range(size)),
        metadata={
            "pool_version": version,
            "gender": args.gender,
            "fixed_race": "unrestricted",
            "fixed_age": "20_29",
            "config_hash": settings_hash,
            "clip_backend": clip.backend,
            "generated": generated,
            "thresholds": settings,
        },
    )
    pool.save()
    print(f"Saved {size} hard-valid candidates to {output} (version {version})")


if __name__ == "__main__":
    main()
