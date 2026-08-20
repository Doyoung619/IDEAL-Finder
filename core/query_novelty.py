from __future__ import annotations

import numpy as np


def inject_novel_global_candidates(
    theta: np.ndarray,
    history: np.ndarray,
    global_mean: np.ndarray,
    global_covariance: np.ndarray,
    *,
    fraction: float,
    pool_size: int,
    seed: int,
    coordinate_clip: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Replace redundant query slots with deterministic farthest global samples."""
    original = np.asarray(theta, dtype=np.float64)
    previous = np.asarray(history, dtype=np.float64)
    if previous.size == 0:
        previous = np.empty((0, original.shape[1]), dtype=np.float64)
    if original.ndim != 2 or previous.ndim != 2:
        raise ValueError("theta and history must be matrices")
    if previous.shape[1] != original.shape[1]:
        raise ValueError("theta and history must have the same dimension")
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("novelty fraction must be in [0, 1)")

    replacement_count = min(
        max(1, int(np.ceil(len(original) * float(fraction)))),
        len(original) - 1,
    ) if fraction > 0.0 and len(original) > 1 and len(previous) else 0
    if replacement_count == 0:
        return original.astype(np.float32), {
            "novelty_slots": 0,
            "novelty_replaced_indices": [],
            "novelty_min_history_distance": None,
        }

    distance_to_history = np.min(
        np.linalg.norm(
            original[:, None, :] - previous[None, :, :], axis=-1
        ),
        axis=1,
    )
    replace_indices = np.argsort(distance_to_history)[:replacement_count]
    replace_set = set(int(index) for index in replace_indices)
    kept = np.asarray(
        [point for index, point in enumerate(original) if index not in replace_set],
        dtype=np.float64,
    )

    random = np.random.default_rng(int(seed) + 32452843)
    candidates = random.multivariate_normal(
        np.asarray(global_mean, dtype=np.float64),
        np.asarray(global_covariance, dtype=np.float64),
        size=max(int(pool_size), replacement_count),
    )
    if coordinate_clip is not None:
        candidates = np.clip(candidates, -coordinate_clip, coordinate_clip)

    references = np.vstack((previous, kept))
    selected: list[np.ndarray] = []
    selected_distances: list[float] = []
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(replacement_count):
        distances = np.min(
            np.linalg.norm(
                candidates[:, None, :] - references[None, :, :], axis=-1
            ),
            axis=1,
        )
        distances[~available] = -np.inf
        index = int(np.argmax(distances))
        chosen = candidates[index]
        selected.append(chosen)
        selected_distances.append(float(distances[index]))
        available[index] = False
        references = np.vstack((references, chosen[None, :]))

    final = original.copy()
    for index, chosen in zip(replace_indices, selected):
        final[int(index)] = chosen
    return final.astype(np.float32), {
        "novelty_slots": replacement_count,
        "novelty_replaced_indices": [int(index) for index in replace_indices],
        "novelty_min_history_distance": float(min(selected_distances)),
    }
