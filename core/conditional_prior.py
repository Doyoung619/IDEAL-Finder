from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DemographicCondition:
    """Operational gender and optional race targets for a conditional prior."""

    gender: str
    race_targets: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.gender not in {"male", "female"}:
            raise ValueError("gender must be 'male' or 'female'")


class ConditionalPCAPrior:
    """Whitened PCA parameterization of a demographic-conditioned W prior."""

    def __init__(
        self,
        condition: DemographicCondition,
        mu_w: np.ndarray,
        components: np.ndarray,
        eigenvalues: np.ndarray,
        explained_variance_ratio: np.ndarray,
        accepted_samples: int,
        thresholds: dict[str, float] | None = None,
        generator_metadata: dict[str, Any] | None = None,
        seed: int = 0,
        theta_mean: np.ndarray | None = None,
        theta_covariance: np.ndarray | None = None,
        covariance_eps: float = 1e-6,
        coordinate_clip: float | None = None,
        sampling_scale: float = 1.0,
        created_at: str | None = None,
    ) -> None:
        self.condition = condition
        self.mu_w = np.asarray(mu_w, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)
        self.eigenvalues = np.maximum(
            np.asarray(eigenvalues, dtype=np.float64), float(covariance_eps)
        )
        self.explained_variance_ratio = np.asarray(
            explained_variance_ratio, dtype=np.float64
        )
        self.accepted_samples = int(accepted_samples)
        self.thresholds = dict(thresholds or {})
        self.generator_metadata = dict(generator_metadata or {})
        self.seed = int(seed)
        self.covariance_eps = float(covariance_eps)
        self.coordinate_clip = (
            None if coordinate_clip is None else float(coordinate_clip)
        )
        self.sampling_scale = float(sampling_scale)
        self.theta_mean = np.asarray(
            theta_mean if theta_mean is not None else np.zeros(self.dimension),
            dtype=np.float64,
        )
        self.theta_covariance = np.asarray(
            theta_covariance
            if theta_covariance is not None
            else np.eye(self.dimension),
            dtype=np.float64,
        )
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self._validate()

    @property
    def dimension(self) -> int:
        """Return the theta-space dimension."""
        if self.components.ndim != 2:
            return 0
        return int(self.components.shape[0])

    @property
    def w_dimension(self) -> int:
        """Return the W-space dimension."""
        return int(self.mu_w.shape[0])

    @property
    def transform_matrix(self) -> np.ndarray:
        """Return the linear map from whitened theta coordinates to W."""
        return (
            self.components.T
            * np.sqrt(self.eigenvalues)[None, :]
            * self.sampling_scale
        )

    def sample_theta(self, n: int, generator=None) -> np.ndarray:
        """Sample theta values reproducibly from the stored Gaussian prior."""
        if n < 0:
            raise ValueError("n must be non-negative")
        random = generator or np.random.default_rng(self.seed)
        samples = random.multivariate_normal(
            self.theta_mean, self.theta_covariance, size=n
        )
        if self.coordinate_clip is not None:
            samples = np.clip(samples, -self.coordinate_clip, self.coordinate_clip)
        return np.asarray(samples, dtype=np.float32).reshape(n, self.dimension)

    def theta_to_w(self, theta: np.ndarray) -> np.ndarray:
        """Map one or more whitened theta vectors into StyleGAN W-space."""
        values, squeezed = self._validate_vectors(theta, self.dimension, "theta")
        if self.coordinate_clip is not None:
            values = np.clip(values, -self.coordinate_clip, self.coordinate_clip)
        result = self.mu_w[None, :] + values @ self.transform_matrix.T
        result = result.astype(np.float32)
        return result[0] if squeezed else result

    def w_to_theta(self, w: np.ndarray) -> np.ndarray:
        """Project one or more W vectors into whitened theta coordinates."""
        values, squeezed = self._validate_vectors(w, self.w_dimension, "w")
        centered = values - self.mu_w[None, :]
        result = (centered @ self.components.T) / (
            np.sqrt(self.eigenvalues)[None, :] * self.sampling_scale
        )
        result = result.astype(np.float32)
        return result[0] if squeezed else result

    def log_prob(self, theta: np.ndarray) -> np.ndarray | float:
        """Evaluate the stored Gaussian theta prior log density."""
        values, squeezed = self._validate_vectors(theta, self.dimension, "theta")
        covariance = 0.5 * (self.theta_covariance + self.theta_covariance.T)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("theta_covariance must be positive definite")
        delta = values - self.theta_mean[None, :]
        solved = np.linalg.solve(covariance, delta.T).T
        log_density = -0.5 * (
            self.dimension * np.log(2.0 * np.pi)
            + log_determinant
            + np.sum(delta * solved, axis=1)
        )
        if self.coordinate_clip is not None:
            outside = np.any(np.abs(values) > self.coordinate_clip + 1e-8, axis=1)
            log_density[outside] = -np.inf
        return float(log_density[0]) if squeezed else log_density

    def save(self, path: str | Path) -> None:
        """Save the prior as a compressed NPZ without Python object payloads."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "condition": asdict(self.condition),
            "accepted_samples": self.accepted_samples,
            "thresholds": self.thresholds,
            "generator_metadata": self.generator_metadata,
            "seed": self.seed,
            "covariance_eps": self.covariance_eps,
            "coordinate_clip": self.coordinate_clip,
            "sampling_scale": self.sampling_scale,
            "created_at": self.created_at,
        }
        np.savez_compressed(
            destination,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            mu_w=self.mu_w.astype(np.float32),
            components=self.components.astype(np.float32),
            eigenvalues=self.eigenvalues.astype(np.float32),
            explained_variance_ratio=self.explained_variance_ratio.astype(
                np.float32
            ),
            theta_mean=self.theta_mean.astype(np.float32),
            theta_covariance=self.theta_covariance.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalPCAPrior":
        """Load and validate a prior artifact without enabling pickle."""
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Conditional prior not found: {source}")
        with np.load(source, allow_pickle=False) as artifact:
            ideal_fields = {
                "mu", "basis", "eigvals", "latent_dim", "clip", "scale", "metadata"
            }
            if ideal_fields.issubset(artifact.files):
                return cls._load_ideal_prior(artifact)
            required = {
                "metadata_json",
                "mu_w",
                "components",
                "eigenvalues",
                "explained_variance_ratio",
                "theta_mean",
                "theta_covariance",
            }
            missing = required.difference(artifact.files)
            if missing:
                raise ValueError(f"Prior artifact is missing fields: {sorted(missing)}")
            metadata = json.loads(str(artifact["metadata_json"].item()))
            if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError("Unsupported conditional prior schema version")
            condition_data = metadata["condition"]
            condition = DemographicCondition(
                gender=condition_data["gender"],
                race_targets=tuple(condition_data["race_targets"]),
            )
            return cls(
                condition=condition,
                mu_w=artifact["mu_w"],
                components=artifact["components"],
                eigenvalues=artifact["eigenvalues"],
                explained_variance_ratio=artifact["explained_variance_ratio"],
                accepted_samples=int(metadata["accepted_samples"]),
                thresholds=metadata.get("thresholds", {}),
                generator_metadata=metadata.get("generator_metadata", {}),
                seed=int(metadata.get("seed", 0)),
                theta_mean=artifact["theta_mean"],
                theta_covariance=artifact["theta_covariance"],
                covariance_eps=float(metadata.get("covariance_eps", 1e-6)),
                coordinate_clip=metadata.get("coordinate_clip"),
                sampling_scale=float(metadata.get("sampling_scale", 1.0)),
                created_at=metadata.get("created_at"),
            )

    @classmethod
    def _load_ideal_prior(cls, artifact) -> "ConditionalPCAPrior":
        """Load the validated StyleGAN3 weighted-PCA artifact from Ideal."""
        metadata = json.loads(str(artifact["metadata"].item()))
        target = str(metadata.get("target", "")).lower()
        if target not in {"female", "male"}:
            raise ValueError("Ideal prior metadata is missing a valid target gender")
        basis = np.asarray(artifact["basis"], dtype=np.float64)
        eigenvalues = np.asarray(artifact["eigvals"], dtype=np.float64)
        latent_dim = int(artifact["latent_dim"])
        if basis.ndim != 2 or basis.shape[1] != latent_dim:
            raise ValueError("Ideal prior basis has an invalid shape")
        total = float(eigenvalues.sum())
        explained = (
            eigenvalues / total
            if total > 0
            else np.full(latent_dim, 1.0 / latent_dim)
        )
        return cls(
            condition=DemographicCondition(target, ("east_asian",)),
            mu_w=np.asarray(artifact["mu"], dtype=np.float64),
            components=basis.T,
            eigenvalues=eigenvalues,
            explained_variance_ratio=explained,
            accepted_samples=int(metadata.get("candidate_count", 1)),
            thresholds={
                str(key): float(value)
                for key, value in metadata.get("thresholds", {}).items()
            },
            generator_metadata={
                "name": metadata.get("generator"),
                "checkpoint": metadata.get("generator_checkpoint"),
                "frozen": metadata.get("generator_frozen", True),
                "source_space": metadata.get("source_space"),
                "target_definition": metadata.get("target_definition"),
                "validation_selection": metadata.get("validation_selection"),
                "artifact_format": "ideal_stylegan3_weighted_pca",
            },
            seed=0,
            coordinate_clip=float(artifact["clip"]),
            sampling_scale=float(artifact["scale"]),
        )

    def _validate(self) -> None:
        arrays = (
            self.mu_w,
            self.components,
            self.eigenvalues,
            self.explained_variance_ratio,
            self.theta_mean,
            self.theta_covariance,
        )
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Conditional prior contains NaN or Inf")
        if self.mu_w.ndim != 1 or self.components.ndim != 2:
            raise ValueError("mu_w must be 1D and components must be 2D")
        if self.components.shape[1] != self.mu_w.shape[0]:
            raise ValueError("PCA components must have shape (d, w_dimension)")
        if self.eigenvalues.shape != (self.dimension,):
            raise ValueError("eigenvalues must have shape (d,)")
        if self.explained_variance_ratio.shape != (self.dimension,):
            raise ValueError("explained_variance_ratio must have shape (d,)")
        if self.theta_mean.shape != (self.dimension,):
            raise ValueError("theta_mean must have shape (d,)")
        if self.theta_covariance.shape != (self.dimension, self.dimension):
            raise ValueError("theta_covariance must have shape (d, d)")
        if self.dimension < 1 or self.accepted_samples < 1:
            raise ValueError("dimension and accepted_samples must be positive")
        if self.sampling_scale <= 0 or not np.isfinite(self.sampling_scale):
            raise ValueError("sampling_scale must be finite and positive")
        if self.coordinate_clip is not None and (
            self.coordinate_clip <= 0 or not np.isfinite(self.coordinate_clip)
        ):
            raise ValueError("coordinate_clip must be finite and positive")
        gram = self.components @ self.components.T
        if not np.allclose(gram, np.eye(self.dimension), atol=1e-4):
            raise ValueError("PCA component rows must be orthonormal")
        if np.linalg.eigvalsh(self.theta_covariance).min() <= 0:
            raise ValueError("theta_covariance must be positive definite")

    @staticmethod
    def _validate_vectors(
        values: np.ndarray, expected_dimension: int, name: str
    ) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=np.float64)
        squeezed = array.ndim == 1
        array = np.atleast_2d(array)
        if array.ndim != 2 or array.shape[1] != expected_dimension:
            raise ValueError(
                f"{name} must have shape ({expected_dimension},) or (n, {expected_dimension})"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return array, squeezed


DemographicConditionalPrior = ConditionalPCAPrior
