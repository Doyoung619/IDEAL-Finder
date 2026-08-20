from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import load_config
from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.demographic_classifier import FairFaceViTDemographicClassifier
from core.generator import create_generator


NORM_BINS = (0.0, 2.0, 3.0, 4.0, 5.0, float("inf"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render and reclassify samples from East-Asian conditional priors."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/persona_study.yaml")
    parser.add_argument("--gender", action="append", choices=("female", "male"))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/east_asian_prior_validation.json")
    args = parser.parse_args()
    if args.samples < 5000:
        raise ValueError("At least 5,000 samples per conditional prior are required")

    config = load_config(args.config, demo_override=False)
    generator = create_generator(config)
    batch_size = int(args.batch_size or config.demographic.batch_size)
    classifier = FairFaceViTDemographicClassifier(
        config.demographic.race_weights,
        config.demographic.gender_weights,
        config.demographic.age_weights,
        devices=list(config.demographic.devices),
        batch_size=batch_size,
    )
    races = tuple(config.demographic.race_targets)
    if races != ("east_asian",):
        raise RuntimeError(f"Expected east_asian condition, got {races}")
    results = []
    for gender_offset, gender in enumerate(args.gender or ("female", "male")):
        prior_path = str(config.conditional_prior.artifact_path).format(
            gender=gender, race="_".join(races)
        )
        prior = ConditionalPCAPrior.load(prior_path)
        expected = DemographicCondition(gender, races)
        if prior.condition != expected:
            raise RuntimeError(
                f"Prior condition {prior.condition} does not match {expected}"
            )
        random = np.random.default_rng(args.seed + gender_offset)
        theta = prior.sample_theta(args.samples, generator=random)
        target_gender_index = classifier.gender_labels.index(gender)
        east_asian_index = classifier.race_labels.index("east_asian")
        age_index = classifier.age_labels.index("20_29")
        gender_probability = []
        race_probability = []
        age_probability = []
        gender_prediction = []
        race_prediction = []
        age_prediction = []
        for start in range(0, args.samples, batch_size):
            values = theta[start : start + batch_size]
            images = generator.decode(prior.theta_to_w(values))
            probabilities = classifier.predict_proba(images)
            gender_values = probabilities["gender"].numpy()
            race_values = probabilities["race"].numpy()
            age_values = probabilities["age"].numpy()
            gender_probability.extend(gender_values[:, target_gender_index])
            race_probability.extend(race_values[:, east_asian_index])
            age_probability.extend(age_values[:, age_index])
            gender_prediction.extend(np.argmax(gender_values, axis=1) == target_gender_index)
            race_prediction.extend(np.argmax(race_values, axis=1) == east_asian_index)
            age_prediction.extend(np.argmax(age_values, axis=1) == age_index)
        gender_prediction = np.asarray(gender_prediction, dtype=bool)
        race_prediction = np.asarray(race_prediction, dtype=bool)
        age_prediction = np.asarray(age_prediction, dtype=bool)
        gender_probability = np.asarray(gender_probability)
        race_probability = np.asarray(race_probability)
        age_probability = np.asarray(age_probability)
        label_all = gender_prediction & race_prediction & age_prediction
        threshold_all = (
            (gender_probability >= float(config.persona_pool.gender_threshold))
            & (race_probability >= float(config.demographic.race_threshold))
            & (age_probability >= float(config.persona_pool.age_20_29_threshold))
        )
        norms = np.linalg.norm(theta, axis=1)
        norm_rows = []
        for lower, upper in zip(NORM_BINS[:-1], NORM_BINS[1:]):
            mask = (norms >= lower) & (norms < upper)
            norm_rows.append(
                {
                    "lower": lower,
                    "upper": None if np.isinf(upper) else upper,
                    "count": int(mask.sum()),
                    "all_predicted_rate": float(label_all[mask].mean()) if mask.any() else None,
                    "all_threshold_rate": float(threshold_all[mask].mean()) if mask.any() else None,
                }
            )
        results.append(
            {
                "gender": gender,
                "prior_path": prior_path,
                "samples": args.samples,
                "east_asian_predicted_rate": float(race_prediction.mean()),
                "age_20_29_predicted_rate": float(age_prediction.mean()),
                "gender_predicted_rate": float(gender_prediction.mean()),
                "all_conditions_predicted_rate": float(label_all.mean()),
                "east_asian_threshold_rate": float((race_probability >= float(config.demographic.race_threshold)).mean()),
                "age_20_29_threshold_rate": float((age_probability >= float(config.persona_pool.age_20_29_threshold)).mean()),
                "gender_threshold_rate": float((gender_probability >= float(config.persona_pool.gender_threshold)).mean()),
                "all_conditions_threshold_rate": float(threshold_all.mean()),
                "theta_norm_bins": norm_rows,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
