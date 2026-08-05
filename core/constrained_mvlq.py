from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.conditional_prior import ConditionalPCAPrior
from core.demographic_classifier import DemographicClassifier
from core.demographic_prior_builder import target_probabilities
from core.mvlq import maximum_variance_line_queries


class NoFeasibleQuerySetError(RuntimeError):
    """Raised when no non-degenerate demographic-feasible line query exists."""

    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(
            "No feasible demographic-constrained MVLQ query set: "
            f"{details}"
        )


@dataclass
class ConstrainedQueryResult:
    """A feasible MVLQ query set and its diagnostic metadata."""

    theta_queries: np.ndarray
    w_queries: np.ndarray
    direction: np.ndarray
    lambda_max: float
    center: np.ndarray
    common_alpha: float
    effective_radius: float
    coefficients: np.ndarray
    gender_probabilities: np.ndarray
    race_probabilities: np.ndarray
    fallback_used: bool
    backtracking_steps: int
    center_source: str

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable query diagnostics."""
        return {
            "direction": self.direction.tolist(),
            "lambda_max": self.lambda_max,
            "center": self.center.tolist(),
            "common_alpha": self.common_alpha,
            "effective_radius": self.effective_radius,
            "coefficients": self.coefficients.tolist(),
            "theta_queries": self.theta_queries.tolist(),
            "w_queries": self.w_queries.tolist(),
            "gender_probabilities": self.gender_probabilities.tolist(),
            "race_probabilities": self.race_probabilities.tolist(),
            "fallback_used": self.fallback_used,
            "backtracking_steps": self.backtracking_steps,
            "center_source": self.center_source,
        }


class DemographicConstrainedMVLQ:
    """Maximum-variance line query with one shared demographic-feasible scale."""

    def __init__(
        self,
        prior: ConditionalPCAPrior,
        generator,
        classifier: DemographicClassifier,
        gender_threshold: float = 0.90,
        race_threshold: float = 0.80,
        backtrack_factor: float = 0.8,
        min_alpha: float = 0.1,
        max_backtracking_steps: int = 15,
        center_candidates: int = 16,
        seed: int = 42,
    ) -> None:
        self.prior = prior
        self.generator = generator
        self.classifier = classifier
        self.gender_threshold = float(gender_threshold)
        self.race_threshold = float(race_threshold)
        self.backtrack_factor = float(backtrack_factor)
        self.min_alpha = float(min_alpha)
        self.max_backtracking_steps = int(max_backtracking_steps)
        self.center_candidates = int(center_candidates)
        self.seed = int(seed)
        if not 0 <= self.gender_threshold <= 1:
            raise ValueError("gender_threshold must be between 0 and 1")
        if not 0 <= self.race_threshold <= 1:
            raise ValueError("race_threshold must be between 0 and 1")
        if not 0 < self.backtrack_factor < 1:
            raise ValueError("backtrack_factor must be between 0 and 1")
        if not 0 < self.min_alpha <= 1:
            raise ValueError("min_alpha must be in (0, 1]")
        if self.max_backtracking_steps < 0 or self.center_candidates < 0:
            raise ValueError("step and candidate counts must be non-negative")

    def propose(
        self,
        posterior_map: np.ndarray,
        posterior_covariance: np.ndarray,
        count: int,
        radius: float,
        posterior_mean: np.ndarray | None = None,
    ) -> ConstrainedQueryResult:
        """Find the best feasible center and largest shared backtracking scale."""
        map_value = self._validate_center(posterior_map, "posterior_map")
        mean_value = self._validate_center(
            posterior_mean if posterior_mean is not None else map_value,
            "posterior_mean",
        )
        covariance = np.asarray(posterior_covariance, dtype=np.float64)
        if covariance.shape != (self.prior.dimension, self.prior.dimension):
            raise ValueError("posterior_covariance has an invalid shape")
        if radius <= 0:
            raise ValueError("radius must be positive")

        centers = self._candidate_centers(map_value, mean_value, covariance)
        feasible_results: list[tuple[float, float, float, ConstrainedQueryResult]] = []
        diagnostics: list[dict[str, Any]] = []
        for center_index, (source, center) in enumerate(centers):
            result, diagnostic = self._search_center(
                center,
                source,
                covariance,
                count,
                radius,
                fallback_used=center_index > 0,
            )
            diagnostics.append(diagnostic)
            if result is None:
                continue
            density = self._gaussian_log_density(center, mean_value, covariance)
            distance = float(np.linalg.norm(center - map_value))
            feasible_results.append(
                (density, result.common_alpha, -distance, result)
            )
        if not feasible_results:
            raise NoFeasibleQuerySetError(
                {
                    "condition": {
                        "gender": self.prior.condition.gender,
                        "race_targets": list(self.prior.condition.race_targets),
                    },
                    "gender_threshold": self.gender_threshold,
                    "race_threshold": self.race_threshold,
                    "last_alpha": diagnostics[-1]["last_alpha"]
                    if diagnostics
                    else None,
                    "centers": diagnostics,
                }
            )
        feasible_results.sort(key=lambda value: value[:3], reverse=True)
        return feasible_results[0][3]

    def _search_center(
        self,
        center: np.ndarray,
        source: str,
        covariance: np.ndarray,
        count: int,
        radius: float,
        fallback_used: bool,
    ) -> tuple[ConstrainedQueryResult | None, dict[str, Any]]:
        alpha = 1.0
        best_min_gender = 0.0
        best_min_race = 0.0
        last_summary: dict[str, Any] = {}
        steps = 0
        while steps <= self.max_backtracking_steps and alpha >= self.min_alpha:
            theta, coefficients, direction, eigenvalue = (
                maximum_variance_line_queries(
                    center, covariance, count, radius, alpha
                )
            )
            w = np.asarray(self.prior.theta_to_w(theta), dtype=np.float32)
            images = self.generator.synthesize_w(w)
            gender, race = target_probabilities(
                self.classifier.predict_proba(images),
                self.classifier,
                self.prior.condition,
            )
            min_gender = float(gender.min())
            min_race = float(race.min())
            best_min_gender = max(best_min_gender, min_gender)
            best_min_race = max(best_min_race, min_race)
            last_summary = {
                "gender": gender.tolist(),
                "race": race.tolist(),
            }
            if min_gender >= self.gender_threshold and min_race >= self.race_threshold:
                return (
                    ConstrainedQueryResult(
                        theta_queries=theta.astype(np.float32),
                        w_queries=w,
                        direction=direction.astype(np.float32),
                        lambda_max=eigenvalue,
                        center=center.astype(np.float32),
                        common_alpha=float(alpha),
                        effective_radius=float(alpha * radius),
                        coefficients=coefficients.astype(np.float32),
                        gender_probabilities=gender.astype(np.float32),
                        race_probabilities=race.astype(np.float32),
                        fallback_used=fallback_used,
                        backtracking_steps=steps,
                        center_source=source,
                    ),
                    {},
                )
            alpha *= self.backtrack_factor
            steps += 1
        return None, {
            "source": source,
            "center": center.tolist(),
            "best_min_gender_probability": best_min_gender,
            "best_min_race_probability": best_min_race,
            "last_alpha": float(alpha),
            "last_query_scores": last_summary,
        }

    def _candidate_centers(
        self,
        posterior_map: np.ndarray,
        posterior_mean: np.ndarray,
        covariance: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        candidates: list[tuple[str, np.ndarray]] = [
            ("posterior_map", posterior_map),
            ("posterior_mean", posterior_mean),
            ("prior_mean", self.prior.theta_mean),
        ]
        if self.center_candidates:
            random = np.random.default_rng(self.seed)
            samples = random.multivariate_normal(
                posterior_mean,
                0.5 * (covariance + covariance.T),
                size=self.center_candidates,
                check_valid="raise",
            )
            order = sorted(
                range(len(samples)),
                key=lambda index: self._gaussian_log_density(
                    samples[index], posterior_mean, covariance
                ),
                reverse=True,
            )
            candidates.extend(
                (f"posterior_candidate_{index}", samples[index])
                for index in order
            )
        unique: list[tuple[str, np.ndarray]] = []
        for source, candidate in candidates:
            value = np.asarray(candidate, dtype=np.float64)
            if not any(np.allclose(value, existing) for _, existing in unique):
                unique.append((source, value))
        return unique

    def _validate_center(self, center: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(center, dtype=np.float64)
        if value.shape != (self.prior.dimension,):
            raise ValueError(f"{name} must have shape ({self.prior.dimension},)")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return value

    @staticmethod
    def _gaussian_log_density(
        point: np.ndarray, mean: np.ndarray, covariance: np.ndarray
    ) -> float:
        jitter = 1e-6 * np.eye(len(point))
        delta = point - mean
        precision = np.linalg.inv(0.5 * (covariance + covariance.T) + jitter)
        return -0.5 * float(delta @ precision @ delta)


ConstrainedMaximumVarianceLineQuery = DemographicConstrainedMVLQ
