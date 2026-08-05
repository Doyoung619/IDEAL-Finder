from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
        """Fit a Laplace posterior while preserving the legacy MVLQ behavior."""
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
