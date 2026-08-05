from __future__ import annotations

import numpy as np

from core.generator import FaceGenerator
from core.pca_utils import LatentProjector


def sample_around_center(
    center: np.ndarray,
    sigma: float,
    count: int,
    projector: LatentProjector,
    generator: FaceGenerator,
    seed: int,
    exploration_fraction: float = 0.25,
    exploration_scale: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)
    exploration_count = min(count, round(count * exploration_fraction))
    local_count = count - exploration_count
    prior = generator.sample_prior(count, seed + 104729)
    local_prior = prior[:local_count]
    blend = random.normal(
        loc=sigma,
        scale=max(0.02, sigma * 0.15),
        size=(local_count, 1),
    ).clip(0.05, 0.45)
    local = (1.0 - blend) * center[None, :] + blend * local_prior
    exploration = prior[local_count:]
    latents = (
        np.vstack([local, exploration])
        if exploration_count
        else local
    ).astype(np.float32)
    random.shuffle(latents, axis=0)
    features = projector.transform(latents)
    return latents, features.astype(np.float32)


def farthest_point_sampling(
    features: np.ndarray,
    count: int,
    seed: int,
    priority_scores: np.ndarray | None = None,
) -> list[int]:
    values = np.atleast_2d(features).astype(np.float32)
    if count >= len(values):
        return list(range(len(values)))
    random = np.random.default_rng(seed)
    if priority_scores is None:
        first = int(random.integers(0, len(values)))
    else:
        first = int(np.argmax(priority_scores))

    selected = [first]
    minimum_distances = np.full(len(values), np.inf, dtype=np.float32)
    while len(selected) < count:
        latest = values[selected[-1]]
        distances = np.linalg.norm(values - latest, axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
        minimum_distances[selected] = -np.inf
        if priority_scores is not None:
            normalized = np.asarray(priority_scores, dtype=np.float32)
            normalized = (normalized - normalized.min()) / (
                np.ptp(normalized) + 1e-8
            )
            objective = minimum_distances * (0.7 + 0.3 * normalized)
        else:
            objective = minimum_distances
        selected.append(int(np.argmax(objective)))
    return selected
