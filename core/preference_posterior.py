from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised in Vercel slim installs.
    torch = None

from core.conditional_prior import ConditionalPCAPrior


@dataclass
class GaussianPreferencePosterior:
    """Laplace Gaussian posterior for M-way distance-based preferences."""

    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    map_estimate: np.ndarray
    covariance: np.ndarray
    beta: float = 1.0
    distance_space: str = "theta"
    metric: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        self.prior_mean = np.asarray(self.prior_mean, dtype=np.float64)
        self.prior_covariance = np.asarray(
            self.prior_covariance, dtype=np.float64
        )
        self.map_estimate = np.asarray(self.map_estimate, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        if self.prior_mean.ndim != 1:
            raise ValueError("prior_mean must be one-dimensional")
        dimension = len(self.prior_mean)
        expected_matrix = (dimension, dimension)
        if self.prior_covariance.shape != expected_matrix:
            raise ValueError("prior_covariance has an invalid shape")
        if self.map_estimate.shape != (dimension,):
            raise ValueError("map_estimate has an invalid shape")
        if self.covariance.shape != expected_matrix:
            raise ValueError("covariance has an invalid shape")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.distance_space not in {"theta", "w"}:
            raise ValueError("distance_space must be 'theta' or 'w'")
        if self.metric is None:
            self.metric = np.eye(dimension, dtype=np.float64)
        else:
            self.metric = np.asarray(self.metric, dtype=np.float64)
        if self.metric.shape != expected_matrix:
            raise ValueError("metric has an invalid shape")
        arrays = (
            self.prior_mean,
            self.prior_covariance,
            self.map_estimate,
            self.covariance,
            self.metric,
        )
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("posterior contains NaN or Inf")

    @property
    def mean(self) -> np.ndarray:
        """Return the current Laplace mean/MAP estimate."""
        return self.map_estimate

    @classmethod
    def initialize_from_prior(
        cls,
        prior: ConditionalPCAPrior,
        beta: float = 1.0,
        distance_space: str = "theta",
    ) -> "GaussianPreferencePosterior":
        """Initialize the posterior from a conditional prior artifact."""
        if distance_space == "theta":
            metric = np.eye(prior.dimension, dtype=np.float64)
        elif distance_space == "w":
            transform = prior.transform_matrix
            metric = transform.T @ transform
        else:
            raise ValueError("distance_space must be 'theta' or 'w'")
        return cls(
            prior_mean=prior.theta_mean.copy(),
            prior_covariance=prior.theta_covariance.copy(),
            map_estimate=prior.theta_mean.copy(),
            covariance=prior.theta_covariance.copy(),
            beta=float(beta),
            distance_space=distance_space,
            metric=metric,
        )

    def choice_probabilities(
        self, queries: np.ndarray, estimate: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute M-way softmax probabilities under the distance likelihood."""
        query_values = self._validate_queries(queries)
        target = self.map_estimate if estimate is None else np.asarray(
            estimate, dtype=np.float64
        )
        if target.shape != self.prior_mean.shape:
            raise ValueError("estimate has an invalid shape")
        linear = query_values @ self.metric @ target
        quadratic = np.einsum(
            "ni,ij,nj->n", query_values, self.metric, query_values
        )
        logits = self.beta * (linear - 0.5 * quadratic)
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        return probabilities / float(probabilities.sum())

    def sample(
        self,
        num_samples: int,
        seed: int = 0,
        device: str | object = "cpu",
        dtype=None,
    ):
        """Draw reparameterized samples from the current Gaussian approximation."""
        if torch is None:
            raise RuntimeError("Torch is required for posterior sampling")
        dtype = dtype or torch.float32
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        target_device = torch.device(device)
        mean = torch.as_tensor(self.map_estimate, dtype=dtype, device=target_device)
        covariance = torch.as_tensor(
            0.5 * (self.covariance + self.covariance.T),
            dtype=dtype,
            device=target_device,
        )
        identity = torch.eye(len(mean), dtype=dtype, device=target_device)
        stable_covariance = None
        for jitter in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            candidate = covariance + jitter * identity
            _, info = torch.linalg.cholesky_ex(candidate)
            if int(info.max().item()) == 0:
                stable_covariance = candidate
                break
        if stable_covariance is None:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            stable_covariance = (
                eigenvectors
                @ torch.diag(eigenvalues.clamp_min(1e-8))
                @ eigenvectors.T
                + 1e-6 * identity
            )
        distribution = torch.distributions.MultivariateNormal(
            mean, covariance_matrix=stable_covariance
        )
        devices = []
        if target_device.type == "cuda":
            devices = [target_device.index or torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if target_device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            return distribution.rsample((num_samples,))

    def update(
        self, queries: np.ndarray, winner_index: int
    ) -> "GaussianPreferencePosterior":
        """Add one preference observation and recompute the Laplace approximation."""
        query_values = self._validate_queries(queries)
        if not 0 <= int(winner_index) < len(query_values):
            raise ValueError("winner_index is outside the query set")
        self.history.append(
            {"queries": query_values.copy(), "winner_index": int(winner_index)}
        )
        self.map_estimate, self.covariance = self.fit(
            prior_mean=self.prior_mean,
            prior_covariance=self.prior_covariance,
            history=self.history,
            initial=self.map_estimate,
            beta=self.beta,
            metric=self.metric,
            jitter=self.jitter,
        )
        return self

    @classmethod
    def fit(
        cls,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray,
        history: list[dict],
        initial: np.ndarray,
        beta: float = 1.0,
        metric: np.ndarray | None = None,
        jitter: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit a Laplace posterior for the shared distance-softmax likelihood."""
        prior_mean = np.asarray(prior_mean, dtype=np.float64)
        prior_covariance = np.asarray(prior_covariance, dtype=np.float64)
        estimate = np.asarray(initial, dtype=np.float64).copy()
        dimension = len(prior_mean)
        identity = np.eye(dimension, dtype=np.float64)
        distance_metric = (
            identity if metric is None else np.asarray(metric, dtype=np.float64)
        )
        prior_precision = np.linalg.inv(prior_covariance + jitter * identity)

        def probabilities(queries: np.ndarray, point: np.ndarray) -> np.ndarray:
            linear = queries @ distance_metric @ point
            quadratic = np.einsum(
                "ni,ij,nj->n", queries, distance_metric, queries
            )
            logits = beta * (linear - 0.5 * quadratic)
            logits -= float(np.max(logits))
            values = np.exp(logits)
            return values / float(values.sum())

        def gradient_precision(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gradient = -prior_precision @ (point - prior_mean)
            precision = prior_precision.copy()
            for observation in history:
                queries = np.asarray(observation["queries"], dtype=np.float64)
                winner = int(observation["winner_index"])
                query_features = beta * (queries @ distance_metric)
                query_probabilities = probabilities(queries, point)
                expectation = query_probabilities @ query_features
                gradient += query_features[winner] - expectation
                centered = query_features - expectation[None, :]
                precision += centered.T @ (
                    query_probabilities[:, None] * centered
                )
            return gradient, precision

        def objective(point: np.ndarray) -> float:
            delta = point - prior_mean
            value = -0.5 * float(delta @ prior_precision @ delta)
            for observation in history:
                queries = np.asarray(observation["queries"], dtype=np.float64)
                winner = int(observation["winner_index"])
                linear = queries @ distance_metric @ point
                quadratic = np.einsum(
                    "ni,ij,nj->n", queries, distance_metric, queries
                )
                logits = beta * (linear - 0.5 * quadratic)
                maximum = float(np.max(logits))
                value += float(logits[winner] - maximum)
                value -= float(np.log(np.exp(logits - maximum).sum()))
            return value

        for _ in range(50):
            gradient, precision = gradient_precision(estimate)
            step = np.linalg.solve(precision + jitter * identity, gradient)
            if float(np.linalg.norm(step)) < 1e-7:
                break
            current_objective = objective(estimate)
            step_scale = 1.0
            while step_scale >= 1e-4:
                candidate = estimate + step_scale * step
                if objective(candidate) >= current_objective:
                    estimate = candidate
                    break
                step_scale *= 0.5
            else:
                break
        _, precision = gradient_precision(estimate)
        covariance = np.linalg.inv(precision + jitter * identity)
        return estimate, 0.5 * (covariance + covariance.T)

    def log_density(self, theta: np.ndarray) -> float:
        """Return the current Gaussian approximation log density up to a constant."""
        point = np.asarray(theta, dtype=np.float64)
        if point.shape != self.map_estimate.shape:
            raise ValueError("theta has an invalid shape")
        precision = np.linalg.inv(
            self.covariance + self.jitter * np.eye(len(point))
        )
        delta = point - self.map_estimate
        return -0.5 * float(delta @ precision @ delta)

    def _validate_queries(self, queries: np.ndarray) -> np.ndarray:
        values = np.asarray(queries, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.prior_mean):
            raise ValueError(
                f"queries must have shape (M, {len(self.prior_mean)})"
            )
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("queries must contain at least two finite rows")
        return values


@dataclass
class ParticleMixturePreferencePosterior:
    """Weighted-particle posterior initialized from an exact two-Gaussian mixture."""

    particles: np.ndarray
    weights: np.ndarray
    component_is_global: np.ndarray
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    beta: float = 1.0
    metric: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)
    resample_threshold: float = 0.12
    rejuvenation_scale: float = 0.05

    def __post_init__(self) -> None:
        self.particles = np.asarray(self.particles, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.component_is_global = np.asarray(self.component_is_global, dtype=bool)
        self.prior_mean = np.asarray(self.prior_mean, dtype=np.float64)
        self.prior_covariance = np.asarray(self.prior_covariance, dtype=np.float64)
        if self.particles.ndim != 2 or len(self.particles) < 2:
            raise ValueError("particles must have shape (N, d) with N >= 2")
        count, dimension = self.particles.shape
        if self.weights.shape != (count,) or self.component_is_global.shape != (count,):
            raise ValueError("particle weights/components have invalid shapes")
        if self.prior_mean.shape != (dimension,) or self.prior_covariance.shape != (dimension, dimension):
            raise ValueError("mixture prior moments have invalid shapes")
        if self.beta <= 0 or np.any(self.weights < 0):
            raise ValueError("beta must be positive and weights non-negative")
        if not all(np.isfinite(value).all() for value in (self.particles, self.weights, self.prior_mean, self.prior_covariance)):
            raise ValueError("particle posterior contains NaN or Inf")
        total = float(self.weights.sum())
        if total <= 0:
            raise ValueError("particle weights must have positive mass")
        self.weights /= total
        self.metric = (
            np.eye(dimension, dtype=np.float64)
            if self.metric is None
            else np.asarray(self.metric, dtype=np.float64)
        )
        if self.metric.shape != (dimension, dimension):
            raise ValueError("metric has an invalid shape")
        self._refresh_moments()
        self.last_resampled = False
        self.ess_before_resample = self.effective_sample_size

    @classmethod
    def from_gaussian_mixture(
        cls,
        local_mean: np.ndarray,
        local_covariance: np.ndarray,
        global_mean: np.ndarray,
        global_covariance: np.ndarray,
        global_weight: float,
        particle_count: int,
        seed: int,
        beta: float = 1.0,
        metric: np.ndarray | None = None,
    ) -> "ParticleMixturePreferencePosterior":
        if not 0.0 < global_weight < 1.0:
            raise ValueError("global_weight must be strictly between zero and one")
        if particle_count < 512:
            raise ValueError("particle_count must be at least 512")
        local_mean = np.asarray(local_mean, dtype=np.float64)
        global_mean = np.asarray(global_mean, dtype=np.float64)
        local_covariance = np.asarray(local_covariance, dtype=np.float64)
        global_covariance = np.asarray(global_covariance, dtype=np.float64)
        random = np.random.default_rng(seed)
        global_count = int(round(particle_count * global_weight))
        local_count = particle_count - global_count
        local = random.multivariate_normal(local_mean, local_covariance, size=local_count)
        global_values = random.multivariate_normal(global_mean, global_covariance, size=global_count)
        particles = np.vstack((local, global_values))
        components = np.concatenate(
            (np.zeros(local_count, dtype=bool), np.ones(global_count, dtype=bool))
        )
        order = random.permutation(particle_count)
        particles = particles[order]
        components = components[order]
        weights = np.where(
            components,
            global_weight / global_count,
            (1.0 - global_weight) / local_count,
        ).astype(np.float64)
        mixture_mean = (1.0 - global_weight) * local_mean + global_weight * global_mean
        local_delta = local_mean - mixture_mean
        global_delta = global_mean - mixture_mean
        mixture_covariance = (
            (1.0 - global_weight)
            * (local_covariance + np.outer(local_delta, local_delta))
            + global_weight
            * (global_covariance + np.outer(global_delta, global_delta))
        )
        return cls(
            particles=particles,
            weights=weights,
            component_is_global=components,
            prior_mean=mixture_mean,
            prior_covariance=mixture_covariance,
            beta=beta,
            metric=metric,
        )

    @property
    def mean(self) -> np.ndarray:
        return self.map_estimate

    @property
    def global_mass(self) -> float:
        return float(self.weights[self.component_is_global].sum())

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.square(self.weights).sum())

    def sample(
        self,
        num_samples: int,
        seed: int = 0,
        device: str | object = "cpu",
        dtype=None,
    ):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        random = np.random.default_rng(seed)
        indices = random.choice(len(self.particles), size=num_samples, replace=True, p=self.weights)
        if torch is None:
            return self.particles[indices].copy()
        dtype = dtype or torch.float32
        return torch.as_tensor(self.particles[indices], dtype=dtype, device=torch.device(device))

    def update(self, queries: np.ndarray, winner_index: int) -> "ParticleMixturePreferencePosterior":
        values = self._validate_queries(queries)
        if not 0 <= int(winner_index) < len(values):
            raise ValueError("winner_index is outside the query set")
        linear = self.particles @ self.metric @ values.T
        quadratic = np.einsum("ni,ij,nj->n", values, self.metric, values)
        logits = self.beta * (2.0 * linear - quadratic[None, :])
        logits -= logits.max(axis=1, keepdims=True)
        likelihood = np.exp(logits)
        likelihood /= likelihood.sum(axis=1, keepdims=True)
        log_weights = np.log(np.clip(self.weights, 1e-300, None))
        log_weights += np.log(np.clip(likelihood[:, int(winner_index)], 1e-300, None))
        log_weights -= float(log_weights.max())
        self.weights = np.exp(log_weights)
        self.weights /= float(self.weights.sum())
        self.history.append({"queries": values.copy(), "winner_index": int(winner_index)})
        self._refresh_moments()
        self.ess_before_resample = self.effective_sample_size
        self.last_resampled = False
        if self.effective_sample_size < self.resample_threshold * len(self.particles):
            self._regularized_resample()
        return self

    def _regularized_resample(self) -> None:
        """Deterministic Liu-West resampling prevents late-round particle collapse."""
        count, dimension = self.particles.shape
        random = np.random.default_rng(104729 * len(self.history) + dimension)
        ancestors = random.choice(count, size=count, replace=True, p=self.weights)
        selected = self.particles[ancestors]
        selected_components = self.component_is_global[ancestors]
        h = float(self.rejuvenation_scale)
        shrinkage = float(np.sqrt(max(1.0 - h * h, 0.0)))
        covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 1e-10, None)))
        noise = random.normal(size=(count, dimension)) @ factor.T
        self.particles = (
            shrinkage * selected
            + (1.0 - shrinkage) * self.map_estimate[None, :]
            + h * noise
        )
        self.component_is_global = selected_components
        self.weights = np.full(count, 1.0 / count, dtype=np.float64)
        self._refresh_moments()
        self.last_resampled = True

    def _refresh_moments(self) -> None:
        self.map_estimate = self.weights @ self.particles
        centered = self.particles - self.map_estimate[None, :]
        covariance = centered.T @ (self.weights[:, None] * centered)
        self.covariance = 0.5 * (covariance + covariance.T)

    def _validate_queries(self, queries: np.ndarray) -> np.ndarray:
        values = np.asarray(queries, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.particles.shape[1]:
            raise ValueError(f"queries must have shape (M, {self.particles.shape[1]})")
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("queries must contain at least two finite rows")
        return values
