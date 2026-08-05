from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.demographic_classifier import DemographicClassifier


@dataclass(frozen=True)
class PriorBuildConfig:
    """Validated controls for demographic conditional-prior construction."""

    condition: DemographicCondition
    latent_dimension: int = 12
    selection_mode: str = "hard"
    num_generator_samples: int = 50_000
    min_accepted_samples: int = 2_000
    batch_size: int = 32
    gender_threshold: float = 0.90
    race_threshold: float = 0.80
    covariance_eps: float = 1e-6
    covariance_shrinkage: float = 0.01
    seed: int = 42
    cache_path: str | None = None

    def __post_init__(self) -> None:
        if self.selection_mode not in {"hard", "soft"}:
            raise ValueError("selection_mode must be 'hard' or 'soft'")
        if self.latent_dimension < 1:
            raise ValueError("latent_dimension must be positive")
        if self.num_generator_samples < 1 or self.min_accepted_samples < 1:
            raise ValueError("sample counts must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.gender_threshold <= 1:
            raise ValueError("gender_threshold must be between 0 and 1")
        if not 0 <= self.race_threshold <= 1:
            raise ValueError("race_threshold must be between 0 and 1")
        if self.covariance_eps <= 0:
            raise ValueError("covariance_eps must be positive")
        if not 0 <= self.covariance_shrinkage <= 1:
            raise ValueError("covariance_shrinkage must be between 0 and 1")


def target_probabilities(
    predictions: dict[str, torch.Tensor],
    classifier: DemographicClassifier,
    condition: DemographicCondition,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract target gender and aggregated target-race probabilities."""
    if set(predictions) != {"gender", "race"}:
        raise ValueError("classifier output must contain gender and race")
    gender = predictions["gender"].detach().cpu().numpy()
    race = predictions["race"].detach().cpu().numpy()
    if gender.ndim != 2 or gender.shape[1] != len(classifier.gender_labels):
        raise ValueError("gender probabilities have an invalid shape")
    if race.ndim != 2 or race.shape[1] != len(classifier.race_labels):
        raise ValueError("race probabilities have an invalid shape")
    if gender.shape[0] != race.shape[0]:
        raise ValueError("gender and race batch sizes differ")
    gender_index = classifier.gender_labels.index(condition.gender)
    race_indices = [
        classifier.race_labels.index(label) for label in condition.race_targets
    ]
    return gender[:, gender_index], race[:, race_indices].sum(axis=1)


def hard_selection_mask(
    gender_probability: np.ndarray,
    race_probability: np.ndarray,
    gender_threshold: float,
    race_threshold: float,
) -> np.ndarray:
    """Return the hard demographic acceptance mask."""
    return (np.asarray(gender_probability) >= gender_threshold) & (
        np.asarray(race_probability) >= race_threshold
    )


def soft_selection_weights(
    gender_probability: np.ndarray, race_probability: np.ndarray
) -> np.ndarray:
    """Return product weights for soft demographic conditioning."""
    weights = np.asarray(gender_probability, dtype=np.float64) * np.asarray(
        race_probability, dtype=np.float64
    )
    return np.clip(weights, 0.0, 1.0)


class DemographicPriorBuilder:
    """Build a conditional PCA prior from generated W latents and predictions."""

    def __init__(self, generator, classifier: DemographicClassifier) -> None:
        self.generator = generator
        self.classifier = classifier

    def build(self, config: PriorBuildConfig) -> ConditionalPCAPrior:
        """Generate or resume a latent bank and fit the requested conditional PCA."""
        w_values, gender_values, race_values = self._collect(config)
        if config.selection_mode == "hard":
            mask = hard_selection_mask(
                gender_values,
                race_values,
                config.gender_threshold,
                config.race_threshold,
            )
            selected = w_values[mask]
            weights = np.ones(len(selected), dtype=np.float64)
        else:
            weights = soft_selection_weights(gender_values, race_values)
            mask = weights > 0
            selected = w_values[mask]
            weights = weights[mask]
        if len(selected) < config.min_accepted_samples:
            raise RuntimeError(
                f"Only {len(selected)} usable samples; required "
                f"{config.min_accepted_samples}. Lower thresholds or sample more latents."
            )
        if config.latent_dimension > min(len(selected) - 1, selected.shape[1]):
            raise ValueError(
                "latent_dimension exceeds the rank supported by accepted samples"
            )
        normalized_weights = weights / float(weights.sum())
        mean = np.sum(selected * normalized_weights[:, None], axis=0)
        centered = selected - mean[None, :]
        covariance = (centered * normalized_weights[:, None]).T @ centered
        trace_scale = float(np.trace(covariance)) / covariance.shape[0]
        covariance = (
            (1.0 - config.covariance_shrinkage) * covariance
            + config.covariance_shrinkage
            * trace_scale
            * np.eye(covariance.shape[0])
        )
        covariance += config.covariance_eps * np.eye(covariance.shape[0])
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (covariance + covariance.T)
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        dimension = config.latent_dimension
        selected_eigenvalues = np.maximum(
            eigenvalues[:dimension], config.covariance_eps
        )
        explained = selected_eigenvalues / max(
            float(np.maximum(eigenvalues, 0).sum()), config.covariance_eps
        )
        components = eigenvectors[:, :dimension].T
        self._orient_components(components)
        metadata: dict[str, Any] = {
            "name": getattr(self.generator, "generator_name", type(self.generator).__name__),
            "w_dimension": int(selected.shape[1]),
            "truncation_psi": float(
                getattr(self.generator, "truncation_psi", 1.0)
            ),
            "selection_mode": config.selection_mode,
            "num_generator_samples": config.num_generator_samples,
        }
        return ConditionalPCAPrior(
            condition=config.condition,
            mu_w=mean,
            components=components,
            eigenvalues=selected_eigenvalues,
            explained_variance_ratio=explained,
            accepted_samples=len(selected),
            thresholds={
                "gender": config.gender_threshold,
                "race": config.race_threshold,
            },
            generator_metadata=metadata,
            seed=config.seed,
            covariance_eps=config.covariance_eps,
        )

    def _collect(
        self, config: PriorBuildConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if config.cache_path and Path(config.cache_path).exists():
            return self._load_cache(config.cache_path, config)
        random = np.random.default_rng(config.seed)
        w_batches: list[np.ndarray] = []
        gender_batches: list[np.ndarray] = []
        race_batches: list[np.ndarray] = []
        for start in range(0, config.num_generator_samples, config.batch_size):
            count = min(
                config.batch_size, config.num_generator_samples - start
            )
            z = random.standard_normal((count, int(self.generator.z_dim))).astype(
                np.float32
            )
            w = np.asarray(self.generator.map_z_to_w(z), dtype=np.float32)
            images = self.generator.synthesize_w(w)
            predictions = self.classifier.predict_proba(images)
            gender, race = target_probabilities(
                predictions, self.classifier, config.condition
            )
            w_batches.append(w)
            gender_batches.append(gender.astype(np.float32))
            race_batches.append(race.astype(np.float32))
        collected = (
            np.concatenate(w_batches),
            np.concatenate(gender_batches),
            np.concatenate(race_batches),
        )
        if config.cache_path:
            destination = Path(config.cache_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                metadata_json=np.asarray(
                    json.dumps(
                        {
                            "gender": config.condition.gender,
                            "race_targets": list(config.condition.race_targets),
                            "num_generator_samples": config.num_generator_samples,
                            "seed": config.seed,
                        },
                        sort_keys=True,
                    )
                ),
                w=collected[0],
                gender_probability=collected[1],
                race_probability=collected[2],
            )
        return collected

    @staticmethod
    def _load_cache(
        path: str | Path, config: PriorBuildConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as cache:
            required = {
                "metadata_json",
                "w",
                "gender_probability",
                "race_probability",
            }
            missing = required.difference(cache.files)
            if missing:
                raise ValueError(f"Latent cache is missing fields: {sorted(missing)}")
            metadata = json.loads(str(cache["metadata_json"].item()))
            expected = {
                "gender": config.condition.gender,
                "race_targets": list(config.condition.race_targets),
                "num_generator_samples": config.num_generator_samples,
                "seed": config.seed,
            }
            if metadata != expected:
                raise ValueError(
                    "Latent cache metadata does not match the requested condition, "
                    "sample count, and seed"
                )
            return (
                np.asarray(cache["w"], dtype=np.float32),
                np.asarray(cache["gender_probability"], dtype=np.float32),
                np.asarray(cache["race_probability"], dtype=np.float32),
            )

    @staticmethod
    def _orient_components(components: np.ndarray) -> None:
        for component in components:
            pivot = int(np.argmax(np.abs(component)))
            if component[pivot] < 0:
                component *= -1
