from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - slim gateway installs do not need torch.
    torch = None


class GlobalExplorationPosteriorView:
    """Acquisition-only mixture that keeps a small amount of global exploration."""

    def __init__(
        self,
        posterior,
        global_mean: np.ndarray,
        global_covariance: np.ndarray,
        rho: float,
    ) -> None:
        self.posterior = posterior
        self.global_mean = np.asarray(global_mean, dtype=np.float64)
        self.global_covariance = np.asarray(global_covariance, dtype=np.float64)
        self.rho = float(rho)
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("exploration rho must be in [0, 1)")

        local_weight = 1.0 - self.rho
        local_mean = np.asarray(posterior.mean, dtype=np.float64)
        self.map_estimate = (
            local_weight * local_mean + self.rho * self.global_mean
        )
        local_delta = local_mean - self.map_estimate
        global_delta = self.global_mean - self.map_estimate
        self.covariance = (
            local_weight
            * (
                np.asarray(posterior.covariance, dtype=np.float64)
                + np.outer(local_delta, local_delta)
            )
            + self.rho
            * (
                self.global_covariance
                + np.outer(global_delta, global_delta)
            )
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.prior_covariance = np.asarray(
            posterior.prior_covariance, dtype=np.float64
        )
        self.beta = float(posterior.beta)
        self.history = list(getattr(posterior, "history", []))

    @property
    def mean(self) -> np.ndarray:
        return self.map_estimate

    def sample(
        self,
        num_samples: int,
        seed: int = 0,
        device: str | object = "cpu",
        dtype=None,
    ):
        random = np.random.default_rng(seed)
        global_count = max(1, int(round(num_samples * self.rho)))
        global_count = min(global_count, num_samples - 1)
        local_count = num_samples - global_count
        indices = random.choice(
            len(self.posterior.particles),
            size=local_count,
            replace=True,
            p=np.asarray(self.posterior.weights, dtype=np.float64),
        )
        local = np.asarray(self.posterior.particles, dtype=np.float64)[indices]
        global_values = random.multivariate_normal(
            self.global_mean, self.global_covariance, size=global_count
        )
        values = np.vstack((local, global_values))[random.permutation(num_samples)]
        if torch is None:
            return values
        return torch.as_tensor(
            values,
            dtype=dtype or torch.float32,
            device=torch.device(device),
        )


def acquisition_posterior(
    posterior,
    global_mean: np.ndarray,
    global_covariance: np.ndarray,
    rho: float,
):
    if float(rho) <= 0.0:
        return posterior
    return GlobalExplorationPosteriorView(
        posterior,
        global_mean,
        global_covariance,
        rho,
    )
