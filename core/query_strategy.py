from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.conditional_prior import ConditionalPCAPrior
from core.preference_posterior import GaussianPreferencePosterior
from core.utils import resolve_device, save_json

try:
    from core.entropy_query import EntropyQueryConfig, EntropyQuerySelector
except ImportError:  # pragma: no cover - exercised in Vercel slim installs.
    EntropyQueryConfig = None
    EntropyQuerySelector = None


@dataclass
class ProposalBatch:
    """Latent queries returned to the shared image-rendering pipeline."""

    latents: np.ndarray
    features: np.ndarray
    backend: str
    roles: list[str] | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class _SelectorConfig:
    posterior_mc_samples: int
    num_restarts: int
    optimization_steps: int
    learning_rate: float
    seed: int
    device: str = "cpu"


@dataclass
class _NumpyEntropyQueryResult:
    query_points: np.ndarray
    mutual_information: float
    predictive_entropy: float
    expected_conditional_entropy: float
    min_pairwise_distance: float
    max_mahalanobis_radius: float
    restart: int
    optimization_steps: int
    initial_mutual_information: float

    def metadata(self) -> dict:
        return {
            "mutual_information": self.mutual_information,
            "predictive_entropy": self.predictive_entropy,
            "expected_conditional_entropy": self.expected_conditional_entropy,
            "min_pairwise_distance": self.min_pairwise_distance,
            "max_mahalanobis_radius": self.max_mahalanobis_radius,
            "restart": self.restart,
            "selected_restart": self.restart,
            "optimization_steps": self.optimization_steps,
            "initial_mutual_information": self.initial_mutual_information,
            "optimizer_backend": "numpy_random_search",
        }


class _NumpyEntropyQuerySelector:
    def __init__(self, config: _SelectorConfig) -> None:
        self.config = config

    def select(
        self,
        posterior,
        prior_covariance: np.ndarray,
        num_options: int,
        seed: int | None = None,
    ) -> _NumpyEntropyQueryResult:
        current_seed = self.config.seed if seed is None else int(seed)
        random = np.random.default_rng(current_seed + 104729)
        center = np.asarray(posterior.map_estimate, dtype=np.float64)
        covariance = 0.5 * (
            np.asarray(posterior.covariance, dtype=np.float64)
            + np.asarray(posterior.covariance, dtype=np.float64).T
        )
        prior_factor = self._stable_cholesky(prior_covariance)
        samples = random.multivariate_normal(
            center,
            covariance + 1e-6 * np.eye(len(center)),
            size=int(self.config.posterior_mc_samples),
        )

        best: _NumpyEntropyQueryResult | None = None
        total_trials = max(
            1,
            self.config.num_restarts * self.config.optimization_steps,
        )
        for trial in range(total_trials):
            raw = random.normal(0.0, 1.0, size=(int(num_options), len(center)))
            norms = np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1.0)
            raw = raw / norms
            queries = center[None, :] + raw @ prior_factor.T
            score, predictive, conditional = self._mutual_information(
                samples,
                queries,
            )
            if len(queries) > 1:
                distances = [
                    np.linalg.norm(queries[i] - queries[j])
                    for i in range(len(queries))
                    for j in range(i + 1, len(queries))
                ]
                minimum_distance = float(min(distances))
            else:
                minimum_distance = 0.0
            result = _NumpyEntropyQueryResult(
                query_points=queries.astype(np.float32),
                mutual_information=float(score),
                predictive_entropy=float(predictive),
                expected_conditional_entropy=float(conditional),
                min_pairwise_distance=minimum_distance,
                max_mahalanobis_radius=float(np.linalg.norm(raw, axis=1).max()),
                restart=trial // max(1, self.config.optimization_steps),
                optimization_steps=int(self.config.optimization_steps),
                initial_mutual_information=float(score),
            )
            if best is None or result.mutual_information > best.mutual_information:
                best = result
        if best is None:
            raise RuntimeError("NumPy entropy query search did not produce a result")
        return best

    @staticmethod
    def _stable_cholesky(covariance: np.ndarray) -> np.ndarray:
        matrix = 0.5 * (
            np.asarray(covariance, dtype=np.float64)
            + np.asarray(covariance, dtype=np.float64).T
        )
        identity = np.eye(matrix.shape[0], dtype=np.float64)
        for jitter in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            try:
                return np.linalg.cholesky(matrix + jitter * identity)
            except np.linalg.LinAlgError:
                continue
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        repaired = (
            eigenvectors
            @ np.diag(np.clip(eigenvalues, 1e-8, None))
            @ eigenvectors.T
        )
        return np.linalg.cholesky(repaired + 1e-6 * identity)

    @staticmethod
    def _mutual_information(
        samples: np.ndarray,
        queries: np.ndarray,
    ) -> tuple[float, float, float]:
        squared_distances = np.sum(
            (samples[:, None, :] - queries[None, :, :]) ** 2,
            axis=-1,
        )
        logits = -0.5 * squared_distances
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        epsilon = np.finfo(np.float64).eps
        predictive = probabilities.mean(axis=0)
        predictive_entropy = -np.sum(
            predictive * np.log(np.maximum(predictive, epsilon))
        )
        conditional = -np.sum(
            probabilities * np.log(np.maximum(probabilities, epsilon)),
            axis=1,
        ).mean()
        return (
            float(predictive_entropy - conditional),
            float(predictive_entropy),
            float(conditional),
        )


class EntropyQueryStrategy:
    """Web-app adapter for direct M-way mutual-information query synthesis."""

    name = "entropy"

    def __init__(
        self,
        generator,
        prior: ConditionalPCAPrior,
        parameters: dict,
    ) -> None:
        self.generator = generator
        self.prior = prior
        config_class = EntropyQueryConfig or _SelectorConfig
        selector_class = EntropyQuerySelector or _NumpyEntropyQuerySelector
        self.selector = selector_class(
            config_class(
                posterior_mc_samples=int(parameters["posterior_mc_samples"]),
                num_restarts=int(parameters["num_restarts"]),
                optimization_steps=int(parameters["optimization_steps"]),
                learning_rate=float(parameters["learning_rate"]),
                seed=int(parameters["seed"]),
                device=resolve_device(str(parameters["device"])),
            )
        )

    def propose(
        self,
        center: np.ndarray,
        sigma: float,
        count: int,
        seed: int,
        state_path: str | None = None,
        round_id: int = 1,
        display_count: int = 2,
    ) -> ProposalBatch:
        """Optimize theta queries directly and map them to decoder-ready W latents."""
        del center, sigma, count
        if not state_path:
            raise ValueError("Entropy Query requires a persistent state path")
        state = self._load_or_initialize_state(state_path)
        posterior = self._posterior_from_state(state)
        result = self.selector.select(
            posterior=posterior,
            prior_covariance=self.prior.theta_covariance,
            num_options=int(display_count),
            seed=int(seed),
        )
        theta_queries = result.query_points.astype(np.float32)
        w_queries = np.asarray(
            self.prior.theta_to_w(theta_queries), dtype=np.float32
        )
        metrics = {
            "round": int(round_id),
            "query_algorithm": self.name,
            "query_center": posterior.map_estimate.tolist(),
            **result.metadata(),
        }
        round_directory = self._round_directory(state_path, round_id)
        round_directory.mkdir(parents=True, exist_ok=True)
        np.save(round_directory / "query_points.npy", theta_queries)
        np.save(round_directory / "query_center.npy", posterior.map_estimate)
        np.save(round_directory / "posterior_mean.npy", posterior.mean)
        np.save(round_directory / "posterior_covariance.npy", posterior.covariance)
        np.save(round_directory / "map_estimate.npy", posterior.map_estimate)
        save_json(round_directory / "entropy_metrics.json", metrics)
        return ProposalBatch(
            latents=w_queries,
            features=theta_queries,
            backend=self.name,
            roles=[f"entropy_query_{index + 1}" for index in range(display_count)],
            metadata=metrics,
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
    ) -> tuple[np.ndarray, float]:
        """Update the shared posterior from the observed M-way choice."""
        del center, sigma
        if not state_path:
            raise ValueError("Entropy Query requires a persistent state path")
        state = self._load_state(state_path)
        posterior = self._posterior_from_state(state)
        theta_queries = np.asarray(
            self.prior.w_to_theta(shown_latents), dtype=np.float64
        )
        winner_theta = np.asarray(
            self.prior.w_to_theta(np.atleast_2d(winner_latent))[0],
            dtype=np.float64,
        )
        winner_index = int(
            np.argmin(np.linalg.norm(theta_queries - winner_theta, axis=1))
        )
        posterior.update(theta_queries, winner_index)
        state["map"] = posterior.map_estimate.astype(np.float32)
        state["covariance"] = posterior.covariance.astype(np.float32)
        state["history"] = posterior.history
        self._save_state(state_path, state)
        round_id = len(posterior.history)
        round_directory = self._round_directory(state_path, round_id)
        round_directory.mkdir(parents=True, exist_ok=True)
        np.save(round_directory / "posterior_mean.npy", posterior.mean)
        np.save(round_directory / "posterior_covariance.npy", posterior.covariance)
        np.save(round_directory / "map_estimate.npy", posterior.map_estimate)
        save_json(
            round_directory / "observed_choice.json",
            {"round": round_id, "observed_choice": winner_index},
        )
        next_center = np.asarray(
            self.prior.theta_to_w(posterior.map_estimate), dtype=np.float32
        )
        return next_center, 1.0

    def _load_or_initialize_state(self, state_path: str) -> dict:
        path = Path(state_path)
        if path.exists():
            return self._load_state(state_path)
        state = {
            "algorithm": self.name,
            "version": 1,
            "dimension": self.prior.dimension,
            "prior_mean": self.prior.theta_mean.astype(np.float32),
            "prior_covariance": self.prior.theta_covariance.astype(np.float32),
            "map": self.prior.theta_mean.astype(np.float32),
            "covariance": self.prior.theta_covariance.astype(np.float32),
            "history": [],
        }
        self._save_state(state_path, state)
        return state

    def _load_state(self, state_path: str) -> dict:
        with Path(state_path).open("rb") as handle:
            state = pickle.load(handle)
        if (
            state.get("algorithm") != self.name
            or int(state.get("dimension", -1)) != self.prior.dimension
        ):
            raise RuntimeError("Stored state is incompatible with Entropy Query")
        return state

    def _posterior_from_state(self, state: dict) -> GaussianPreferencePosterior:
        return GaussianPreferencePosterior(
            prior_mean=np.asarray(state["prior_mean"], dtype=np.float64),
            prior_covariance=np.asarray(
                state["prior_covariance"], dtype=np.float64
            ),
            map_estimate=np.asarray(state["map"], dtype=np.float64),
            covariance=np.asarray(state["covariance"], dtype=np.float64),
            beta=1.0,
            distance_space="theta",
            metric=np.eye(self.prior.dimension),
            history=list(state["history"]),
        )

    @staticmethod
    def _round_directory(state_path: str, round_id: int) -> Path:
        return Path(state_path).parent / f"round_{int(round_id):02d}"

    @staticmethod
    def _save_state(state_path: str, state: dict) -> None:
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(state, handle)


def create_query_strategy(
    config,
    generator,
    projector=None,
    mode: str | None = None,
    parameters: dict | None = None,
    conditional_prior: ConditionalPCAPrior | None = None,
):
    """Create the sole user-facing query strategy."""
    del projector
    selected = mode or config.query.algorithm
    if selected != "entropy":
        raise ValueError("Only the entropy query algorithm is currently supported.")
    if conditional_prior is None:
        raise ValueError("Entropy Query requires a conditional latent prior")
    if config.query.constraint != "prior_ellipsoid":
        raise ValueError("Entropy Query requires the prior_ellipsoid constraint")
    values = {
        "posterior_mc_samples": config.query.posterior_mc_samples,
        "num_restarts": config.query.num_restarts,
        "optimization_steps": config.query.optimization_steps,
        "learning_rate": config.query.learning_rate,
        "seed": config.query.seed,
        "device": config.generator.device,
    }
    values.update(parameters or {})
    return EntropyQueryStrategy(generator, conditional_prior, values)
