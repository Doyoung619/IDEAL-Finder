from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QueryDiversityConfig:
    enabled: bool = False
    latent_min_distance: float = 0.15
    image_similarity_threshold: float | None = None
    max_retries: int = 3
    perturbation_initial: float = 0.10
    perturbation_max: float = 0.40
    max_relative_ig_loss: float = 0.05
    radius_expansion_factors: tuple[float, ...] = (1.1, 1.2, 1.35)
    orthogonal_fallback: bool = True


def pairwise_diagnostics(
    theta: np.ndarray, embeddings: np.ndarray
) -> dict[str, object]:
    values = np.asarray(theta, dtype=np.float64)
    features = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or features.ndim != 2 or len(values) != len(features):
        raise ValueError("theta and embeddings must be aligned matrices")
    if len(values) < 2:
        return {
            "min_pairwise_theta_distance": 0.0,
            "max_pairwise_image_similarity": 1.0,
            "mean_pairwise_image_similarity": 1.0,
            "pairwise_image_similarities": [1.0],
        }
    pairs = np.triu_indices(len(values), k=1)
    distances = np.linalg.norm(
        values[:, None, :] - values[None, :, :], axis=-1
    )[pairs]
    similarities = (features @ features.T)[pairs]
    return {
        "min_pairwise_theta_distance": float(np.min(distances)),
        "max_pairwise_image_similarity": float(np.max(similarities)),
        "mean_pairwise_image_similarity": float(np.mean(similarities)),
        "pairwise_image_similarities": similarities.tolist(),
    }


def ensure_query_diversity(
    theta: np.ndarray,
    posterior_covariance: np.ndarray,
    prior,
    generator,
    image_embedder,
    config: QueryDiversityConfig,
    posterior_particles: np.ndarray | None = None,
    posterior_weights: np.ndarray | None = None,
    beta: float = 1.0,
    algorithm: str = "entropy",
) -> tuple[np.ndarray, dict]:
    """Apply a minimal, secondary-direction correction to perceptual duplicates."""
    original = np.asarray(theta, dtype=np.float64)
    final = original.copy()
    if prior.coordinate_clip is not None:
        final = np.clip(final, -prior.coordinate_clip, prior.coordinate_clip)
    covariance = np.asarray(posterior_covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    directions = eigenvectors[:, np.argsort(eigenvalues)[::-1][1:]]

    def measure(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        w = np.asarray(prior.theta_to_w(values), dtype=np.float32)
        images = generator.decode(w)
        embeddings = image_embedder.encode_images(images)
        return embeddings, pairwise_diagnostics(values, embeddings)

    embeddings, before = measure(final)
    corrected: set[int] = set()
    maximum_applied = 0.0
    retries = 0
    radius_expansion_factor = 1.0
    radius_candidates: list[dict] = []
    threshold = config.image_similarity_threshold
    initial_information = (
        expected_information_gain(
            posterior_particles, posterior_weights, final, beta
        )
        if posterior_particles is not None and posterior_weights is not None
        else None
    )
    if (
        config.enabled
        and threshold is not None
        and algorithm == "rc_mlq"
        and before["max_pairwise_image_similarity"] > threshold
        and posterior_particles is not None
        and posterior_weights is not None
    ):
        center = np.average(
            np.asarray(posterior_particles, dtype=np.float64),
            axis=0,
            weights=np.asarray(posterior_weights, dtype=np.float64),
        )
        primary = eigenvectors[:, int(np.argmax(eigenvalues))]
        scalar = (final - center[None, :]) @ primary
        for factor in config.radius_expansion_factors:
            candidate = center[None, :] + (float(factor) * scalar)[:, None] * primary
            if prior.coordinate_clip is not None and np.any(
                np.abs(candidate) > prior.coordinate_clip
            ):
                radius_candidates.append(
                    {
                        "factor": float(factor),
                        "accepted": False,
                        "reason": "outside_prior_boundary",
                    }
                )
                continue
            candidate_embeddings, candidate_metrics = measure(candidate)
            information = expected_information_gain(
                posterior_particles, posterior_weights, candidate, beta
            )
            relative_loss = (
                (initial_information - information) / initial_information
                if initial_information and initial_information > 1e-12
                else 0.0
            )
            radius_candidates.append(
                {
                    "factor": float(factor),
                    "accepted": False,
                    "max_pairwise_image_similarity": candidate_metrics[
                        "max_pairwise_image_similarity"
                    ],
                    "information_gain": information,
                    "relative_information_gain_loss": relative_loss,
                }
            )
            if (
                candidate_metrics["max_pairwise_image_similarity"] <= threshold
                and relative_loss <= config.max_relative_ig_loss
            ):
                final = candidate
                embeddings = candidate_embeddings
                corrected.update(range(len(final)))
                maximum_applied = float(
                    np.linalg.norm(final - original, axis=1).max()
                )
                radius_expansion_factor = float(factor)
                radius_candidates[-1]["accepted"] = True
                break
    allow_orthogonal = algorithm != "rc_mlq" or config.orthogonal_fallback
    if (
        config.enabled
        and threshold is not None
        and not corrected
        and allow_orthogonal
        and directions.shape[1] > 0
    ):
        for retry in range(config.max_retries):
            similarity = embeddings @ embeddings.T
            duplicates: list[int] = []
            for right in range(1, len(final)):
                if any(
                    similarity[left, right] > threshold
                    for left in range(right)
                ):
                    duplicates.append(right)
            if not duplicates:
                break
            magnitude = min(
                config.perturbation_initial * (retry + 1),
                config.perturbation_max,
            )
            for offset, index in enumerate(duplicates):
                direction = directions[:, offset % directions.shape[1]]
                sign = -1.0 if (index + retry) % 2 else 1.0
                candidate = original[index] + sign * magnitude * direction
                if prior.coordinate_clip is not None:
                    candidate = np.clip(
                        candidate, -prior.coordinate_clip, prior.coordinate_clip
                    )
                displacement = candidate - original[index]
                norm = float(np.linalg.norm(displacement))
                if norm > config.perturbation_max:
                    candidate = (
                        original[index]
                        + displacement * (config.perturbation_max / norm)
                    )
                    norm = config.perturbation_max
                final[index] = candidate
                corrected.add(index)
                maximum_applied = max(maximum_applied, norm)
            retries = retry + 1
            embeddings, _ = measure(final)
    _, after = measure(final) if corrected else (embeddings, before)
    metadata = {
        **after,
        "pre_guard_min_pairwise_theta_distance": before[
            "min_pairwise_theta_distance"
        ],
        "pre_guard_max_pairwise_image_similarity": before[
            "max_pairwise_image_similarity"
        ],
        "pre_guard_mean_pairwise_image_similarity": before[
            "mean_pairwise_image_similarity"
        ],
        "diversity_guard_triggered": bool(corrected),
        "num_corrected_queries": len(corrected),
        "max_applied_perturbation": maximum_applied,
        "diversity_guard_retries": retries,
        "radius_expansion_factor": radius_expansion_factor,
        "radius_expansion_candidates": radius_candidates,
        "orthogonal_fallback_used": (
            algorithm == "rc_mlq"
            and bool(corrected)
            and radius_expansion_factor == 1.0
        ),
        "original_theta": original.tolist(),
        "final_theta": final.tolist(),
    }
    if posterior_particles is not None and posterior_weights is not None:
        before_information = expected_information_gain(
            posterior_particles, posterior_weights, original, beta
        )
        after_information = expected_information_gain(
            posterior_particles, posterior_weights, final, beta
        )
        metadata.update(
            {
                "information_gain_before_correction": before_information,
                "information_gain_after_correction": after_information,
                "relative_information_gain_loss": (
                    (before_information - after_information) / before_information
                    if before_information > 1e-12
                    else 0.0
                ),
            }
        )
    return final.astype(np.float32), metadata


def expected_information_gain(
    particles: np.ndarray,
    weights: np.ndarray,
    queries: np.ndarray,
    beta: float,
) -> float:
    values = np.asarray(particles, dtype=np.float64)
    probabilities_weight = np.asarray(weights, dtype=np.float64)
    probabilities_weight /= probabilities_weight.sum()
    query_values = np.asarray(queries, dtype=np.float64)
    squared = np.sum(
        (values[:, None, :] - query_values[None, :, :]) ** 2, axis=-1
    )
    logits = -float(beta) * squared
    logits -= logits.max(axis=1, keepdims=True)
    choice = np.exp(logits)
    choice /= choice.sum(axis=1, keepdims=True)
    predictive = probabilities_weight @ choice
    epsilon = np.finfo(np.float64).eps
    predictive_entropy = -np.sum(
        predictive * np.log(np.clip(predictive, epsilon, None))
    )
    conditional = -np.sum(
        probabilities_weight
        * np.sum(choice * np.log(np.clip(choice, epsilon, None)), axis=1)
    )
    return float(predictive_entropy - conditional)
