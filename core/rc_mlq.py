from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RCMLQConfig:
    resolution_min: float = 0.2
    resolution_max: float = 6.0
    resolution_steps: int = 120
    posterior_samples: int = 4096
    beta: float = 1.0
    coordinate_clip: float | None = None

    def __post_init__(self) -> None:
        if self.resolution_min <= 0 or self.resolution_max <= self.resolution_min:
            raise ValueError("RC-MLQ resolution bounds must be positive and ordered")
        if self.resolution_steps < 8 or self.posterior_samples < 512:
            raise ValueError("RC-MLQ grids/samples are too small")
        if self.beta <= 0:
            raise ValueError("RC-MLQ beta must be positive")


@dataclass(frozen=True)
class RCMLQResult:
    query_points: np.ndarray
    direction: np.ndarray
    quantile_template: np.ndarray
    projected_quantiles: np.ndarray
    posterior_center_offset: float
    posterior_width: float
    physical_resolution: float
    relative_resolution: float
    mutual_information: float
    fisher_information: float
    principal_variance: float
    resolution_grid_size: int
    min_pairwise_distance: float

    def metadata(self) -> dict[str, Any]:
        return {
            "direction": self.direction.tolist(),
            "quantile_template": self.quantile_template.tolist(),
            "projected_quantiles": self.projected_quantiles.tolist(),
            "posterior_center_offset": self.posterior_center_offset,
            "posterior_width": self.posterior_width,
            "physical_resolution": self.physical_resolution,
            "relative_resolution": self.relative_resolution,
            "mutual_information": self.mutual_information,
            "fisher_information": self.fisher_information,
            "principal_variance": self.principal_variance,
            "resolution_grid_size": self.resolution_grid_size,
            "resolution_method": "exact_eig_on_fixed_physical_grid",
            "min_pairwise_distance": self.min_pairwise_distance,
        }


class RCMLQSelector:
    """Resolution-Calibrated Multiway Line Query from the supplied paper."""

    def __init__(self, config: RCMLQConfig) -> None:
        self.config = config

    def select(self, posterior, num_options: int, seed: int = 0) -> RCMLQResult:
        if num_options < 2:
            raise ValueError("num_options must be at least two")
        mean = np.asarray(posterior.mean, dtype=np.float64)
        covariance = np.asarray(posterior.covariance, dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        principal_index = int(np.argmax(eigenvalues))
        principal_variance = max(float(eigenvalues[principal_index]), 1e-10)
        direction = np.asarray(eigenvectors[:, principal_index], dtype=np.float64)
        nonzero = np.flatnonzero(np.abs(direction) > 1e-10)
        if len(nonzero) and direction[nonzero[0]] < 0:
            direction = -direction

        sampled = posterior.sample(
            self.config.posterior_samples, seed=seed, device="cpu"
        )
        if hasattr(sampled, "detach"):
            sampled = sampled.detach().cpu().numpy()
        samples = np.asarray(sampled, dtype=np.float64)
        projected = (samples - mean[None, :]) @ direction
        probabilities = (np.arange(num_options, dtype=np.float64) + 0.5) / num_options
        quantiles = np.quantile(projected, probabilities)
        median = float(np.quantile(projected, 0.5))
        width = float(np.sqrt(principal_variance))
        template = (quantiles - median) / max(width, 1e-8)
        if float(np.ptp(template)) < 1e-6:
            template = np.linspace(-1.0, 1.0, num_options)

        grid = np.geomspace(
            self.config.resolution_min,
            self.config.resolution_max,
            self.config.resolution_steps,
        )
        best: tuple[float, float, float, np.ndarray] | None = None
        for resolution in grid:
            scalar_locations = median + float(resolution) * template
            points = mean[None, :] + scalar_locations[:, None] * direction[None, :]
            if self.config.coordinate_clip is not None and np.any(
                np.abs(points) > self.config.coordinate_clip
            ):
                continue
            information = self._mutual_information(
                projected, scalar_locations, float(posterior.beta)
            )
            fisher = self._fisher_information(
                scalar_locations, float(posterior.beta)
            )
            key = (information, fisher, -float(resolution))
            if best is None or key > best[:3]:
                best = (information, fisher, -float(resolution), points)
        if best is None:
            raise RuntimeError("No RC-MLQ physical resolution fits the latent support")

        information, fisher, negative_resolution, points = best
        resolution = -negative_resolution
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        min_distance = float(np.min(distances))
        return RCMLQResult(
            query_points=points.astype(np.float32),
            direction=direction.astype(np.float32),
            quantile_template=template.astype(np.float32),
            projected_quantiles=quantiles.astype(np.float32),
            posterior_center_offset=median,
            posterior_width=width,
            physical_resolution=resolution,
            relative_resolution=resolution / max(width, 1e-8),
            mutual_information=information,
            fisher_information=fisher,
            principal_variance=principal_variance,
            resolution_grid_size=len(grid),
            min_pairwise_distance=min_distance,
        )

    def _choice_probabilities(
        self, projected_samples: np.ndarray, locations: np.ndarray, beta: float
    ) -> np.ndarray:
        logits = beta * (
            2.0 * projected_samples[:, None] * locations[None, :]
            - np.square(locations)[None, :]
        )
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _mutual_information(
        self, projected_samples: np.ndarray, locations: np.ndarray, beta: float
    ) -> float:
        probabilities = self._choice_probabilities(projected_samples, locations, beta)
        predictive = probabilities.mean(axis=0)
        epsilon = np.finfo(np.float64).eps
        predictive_entropy = -np.sum(predictive * np.log(np.clip(predictive, epsilon, None)))
        conditional_entropy = -np.mean(
            np.sum(probabilities * np.log(np.clip(probabilities, epsilon, None)), axis=1)
        )
        return float(predictive_entropy - conditional_entropy)

    def _fisher_information(self, locations: np.ndarray, beta: float) -> float:
        logits = -beta * np.square(locations)
        logits -= float(logits.max())
        probabilities = np.exp(logits)
        probabilities /= float(probabilities.sum())
        mean = float(probabilities @ locations)
        variance = float(probabilities @ np.square(locations - mean))
        return float(4.0 * beta**2 * variance)
