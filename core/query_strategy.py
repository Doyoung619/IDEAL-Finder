from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.conditional_prior import ConditionalPCAPrior
from core.preference_posterior import ParticleMixturePreferencePosterior
from core.rc_mlq import RCMLQConfig, RCMLQSelector
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
    output_spread_scale: float = 1.0


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
        self.warm_start_required = bool(parameters.get("warm_start_required", False))
        self.global_mixture_weight = float(parameters.get("global_mixture_weight", 0.45))
        self.global_covariance_scale = float(parameters.get("global_covariance_scale", 1.0))
        self.mixture_particle_count = int(parameters.get("mixture_particle_count", 8192))
        self.posterior_beta = float(parameters.get("beta", 1.0))
        self.adaptive_beta_enabled = bool(parameters.get("adaptive_beta_enabled", True))
        self.beta_min = float(parameters.get("beta_min", 0.35))
        self.beta_max = float(parameters.get("beta_max", 2.5))
        self.beta_smoothing = float(parameters.get("beta_smoothing", 0.30))
        self.posterior_seed = int(parameters.get("posterior_seed", parameters["seed"]))
        self.selector = selector_class(
            config_class(
                posterior_mc_samples=int(parameters["posterior_mc_samples"]),
                num_restarts=int(parameters["num_restarts"]),
                optimization_steps=int(parameters["optimization_steps"]),
                learning_rate=float(parameters["learning_rate"]),
                seed=int(parameters["seed"]),
                device=resolve_device(str(parameters["device"])),
                output_spread_scale=float(parameters.get("output_spread_scale", 1.4)),
            )
        )

    def initialize_state(
        self,
        state_path: str,
        initial_mean: np.ndarray,
        covariance_scale: float = 1.0,
        metadata: dict | None = None,
    ) -> dict:
        """Create p0=(1-lambda)N(persona,Sigma_local)+lambda*p_global."""
        path = Path(state_path)
        mean = np.asarray(initial_mean, dtype=np.float64)
        if mean.shape != (self.prior.dimension,) or not np.isfinite(mean).all():
            raise ValueError(
                f"initial_mean must be finite with shape ({self.prior.dimension},)"
            )
        scale = float(covariance_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("covariance_scale must be positive and finite")
        local_covariance = np.asarray(self.prior.theta_covariance, dtype=np.float64) * scale
        global_covariance = (
            np.asarray(self.prior.theta_covariance, dtype=np.float64)
            * self.global_covariance_scale
        )
        if path.exists():
            state = self._load_state(state_path)
            mixture = state.get("mixture", {})
            if not np.allclose(mixture.get("local_mean", []), mean) or not np.isclose(
                float(state.get("prior_covariance_scale", 1.0)), scale
            ):
                raise RuntimeError("Stored state has a different persona warm start")
            return state
        posterior = ParticleMixturePreferencePosterior.from_gaussian_mixture(
            local_mean=mean,
            local_covariance=local_covariance,
            global_mean=np.asarray(self.prior.theta_mean, dtype=np.float64),
            global_covariance=global_covariance,
            global_weight=self.global_mixture_weight,
            particle_count=self.mixture_particle_count,
            seed=self.posterior_seed,
            beta=self.posterior_beta,
            metric=np.eye(self.prior.dimension),
        )
        state = {
            "algorithm": self.name,
            "version": 3,
            "dimension": self.prior.dimension,
            "prior_mean": posterior.prior_mean.astype(np.float32),
            "prior_covariance": posterior.prior_covariance.astype(np.float32),
            "prior_covariance_scale": scale,
            "map": posterior.mean.astype(np.float32),
            "covariance": posterior.covariance.astype(np.float32),
            "particles": posterior.particles.astype(np.float32),
            "weights": posterior.weights,
            "component_is_global": posterior.component_is_global,
            "beta": self.posterior_beta,
            "beta_history": [],
            "history": [],
            "mixture": {
                "local_mean": mean.astype(np.float32),
                "local_covariance": local_covariance.astype(np.float32),
                "global_mean": self.prior.theta_mean.astype(np.float32),
                "global_covariance": global_covariance.astype(np.float32),
                "global_weight": self.global_mixture_weight,
                "particle_count": self.mixture_particle_count,
            },
            "initialization": {
                "type": "persona_warm_start",
                **(metadata or {}),
            },
        }
        self._save_state(state_path, state)
        return state

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
            prior_covariance=posterior.prior_covariance,
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
            "global_component_mass": posterior.global_mass,
            "effective_sample_size": posterior.effective_sample_size,
            "beta": posterior.beta,
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
        preference_rating: int | None = None,
        difficulty_rating: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Update the shared posterior from the observed M-way choice."""
        del center, sigma
        if not state_path:
            raise ValueError("Entropy Query requires a persistent state path")
        state = self._load_state(state_path)
        posterior = self._posterior_from_state(state)
        previous_beta = posterior.beta
        posterior.beta = self._adapt_beta(
            previous_beta, preference_rating, difficulty_rating
        )
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
        state["particles"] = posterior.particles.astype(np.float32)
        state["weights"] = posterior.weights
        state["component_is_global"] = posterior.component_is_global
        state["beta"] = posterior.beta
        state.setdefault("beta_history", []).append(
            {
                "round": len(posterior.history),
                "previous_beta": previous_beta,
                "beta": posterior.beta,
                "preference_rating": preference_rating,
                "difficulty_rating": difficulty_rating,
            }
        )
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
            {
                "round": round_id,
                "observed_choice": winner_index,
                "global_component_mass": posterior.global_mass,
                "effective_sample_size": posterior.effective_sample_size,
                "ess_before_resample": posterior.ess_before_resample,
                "particle_resampled": posterior.last_resampled,
                "previous_beta": previous_beta,
                "beta": posterior.beta,
            },
        )
        next_center = np.asarray(
            self.prior.theta_to_w(posterior.map_estimate), dtype=np.float32
        )
        return next_center, 1.0

    def _adapt_beta(
        self,
        current_beta: float,
        preference_rating: int | None,
        difficulty_rating: int | None,
    ) -> float:
        """Bounded log-space update from perceived closeness and choice difficulty."""
        if (
            not self.adaptive_beta_enabled
            or preference_rating is None
            or difficulty_rating is None
        ):
            return float(current_beta)
        closeness = np.clip((float(preference_rating) - 1.0) / 9.0, 0.0, 1.0)
        ease = np.clip((7.0 - float(difficulty_rating)) / 6.0, 0.0, 1.0)
        reliability = 0.35 * closeness + 0.65 * ease
        target = self.beta_min * (self.beta_max / self.beta_min) ** reliability
        updated = np.exp(
            (1.0 - self.beta_smoothing) * np.log(float(current_beta))
            + self.beta_smoothing * np.log(float(target))
        )
        return float(np.clip(updated, self.beta_min, self.beta_max))

    def _load_or_initialize_state(self, state_path: str) -> dict:
        path = Path(state_path)
        if path.exists():
            return self._load_state(state_path)
        if self.warm_start_required:
            raise RuntimeError(
                "Persona warm start is required before the experiment can begin."
            )
        return self.initialize_state(
            state_path,
            initial_mean=self.prior.theta_mean,
            covariance_scale=1.0,
            metadata={"type": "conditional_prior_fallback"},
        )

    def _load_state(self, state_path: str) -> dict:
        with Path(state_path).open("rb") as handle:
            state = pickle.load(handle)
        if (
            state.get("algorithm") != self.name
            or int(state.get("dimension", -1)) != self.prior.dimension
        ):
            raise RuntimeError(f"Stored state is incompatible with {self.name}")
        if int(state.get("version", -1)) != 3:
            raise RuntimeError("Stored state predates the global-mixture posterior")
        return state

    def _posterior_from_state(self, state: dict) -> ParticleMixturePreferencePosterior:
        return ParticleMixturePreferencePosterior(
            particles=np.asarray(state["particles"], dtype=np.float64),
            weights=np.asarray(state["weights"], dtype=np.float64),
            component_is_global=np.asarray(state["component_is_global"], dtype=bool),
            prior_mean=np.asarray(state["prior_mean"], dtype=np.float64),
            prior_covariance=np.asarray(
                state["prior_covariance"], dtype=np.float64
            ),
            beta=float(state.get("beta", self.posterior_beta)),
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


class RCMLQStrategy(EntropyQueryStrategy):
    """Adapter for the paper's resolution-calibrated multiway line query."""

    name = "rc_mlq"

    def __init__(self, generator, prior: ConditionalPCAPrior, parameters: dict) -> None:
        super().__init__(generator, prior, parameters)
        self.rc_selector = RCMLQSelector(
            RCMLQConfig(
                resolution_min=float(parameters.get("resolution_min", 0.2)),
                resolution_max=float(parameters.get("resolution_max", 6.0)),
                resolution_steps=int(parameters.get("resolution_steps", 120)),
                posterior_samples=int(parameters.get("rc_posterior_samples", 4096)),
                beta=self.posterior_beta,
                coordinate_clip=parameters.get("coordinate_clip"),
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
        del center, sigma, count
        if not state_path:
            raise ValueError("RC-MLQ requires a persistent state path")
        state = self._load_or_initialize_state(state_path)
        posterior = self._posterior_from_state(state)
        result = self.rc_selector.select(posterior, int(display_count), int(seed))
        theta_queries = result.query_points.astype(np.float32)
        w_queries = np.asarray(self.prior.theta_to_w(theta_queries), dtype=np.float32)
        metrics = {
            "round": int(round_id),
            "query_algorithm": self.name,
            "query_center": posterior.mean.tolist(),
            "global_component_mass": posterior.global_mass,
            "effective_sample_size": posterior.effective_sample_size,
            "beta": posterior.beta,
            **result.metadata(),
        }
        round_directory = self._round_directory(state_path, round_id)
        round_directory.mkdir(parents=True, exist_ok=True)
        np.save(round_directory / "query_points.npy", theta_queries)
        np.save(round_directory / "query_center.npy", posterior.mean)
        np.save(round_directory / "posterior_mean.npy", posterior.mean)
        np.save(round_directory / "posterior_covariance.npy", posterior.covariance)
        np.save(round_directory / "map_estimate.npy", posterior.map_estimate)
        save_json(round_directory / "rc_mlq_metrics.json", metrics)
        return ProposalBatch(
            latents=w_queries,
            features=theta_queries,
            backend=self.name,
            roles=[f"rc_mlq_query_{index + 1}" for index in range(display_count)],
            metadata=metrics,
        )


def create_query_strategy(
    config,
    generator,
    projector=None,
    mode: str | None = None,
    parameters: dict | None = None,
    conditional_prior: ConditionalPCAPrior | None = None,
):
    """Create one of the two experiment query strategies."""
    del projector
    selected = mode or config.query.algorithm
    if selected not in {"entropy", "rc_mlq"}:
        raise ValueError(f"Unsupported query algorithm: {selected}")
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
        "warm_start_required": bool(config.persona.required),
        "output_spread_scale": float(config.query.entropy_output_spread_scale),
        "global_mixture_weight": float(config.persona.global_mixture_weight),
        "global_covariance_scale": float(config.persona.global_covariance_scale),
        "mixture_particle_count": int(config.persona.mixture_particle_count),
        "beta": float(config.query.beta),
        "adaptive_beta_enabled": bool(config.query.adaptive_beta_enabled),
        "beta_min": float(config.query.beta_min),
        "beta_max": float(config.query.beta_max),
        "beta_smoothing": float(config.query.beta_smoothing),
        "resolution_min": float(config.query.rc_resolution_min),
        "resolution_max": float(config.query.rc_resolution_max),
        "resolution_steps": int(config.query.rc_resolution_steps),
        "rc_posterior_samples": int(config.query.rc_posterior_samples),
    }
    values.update(parameters or {})
    strategy_class = EntropyQueryStrategy if selected == "entropy" else RCMLQStrategy
    return strategy_class(generator, conditional_prior, values)
