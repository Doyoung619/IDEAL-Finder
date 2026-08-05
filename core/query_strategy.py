from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.generator import FaceGenerator
from core.benchmark_strategies import (
    AxisPairSearchStrategy,
    BanditBOStrategy,
    FastDirectLatentESStrategy,
    LinearPreferenceUCBStrategy,
    QEUBOStrategy,
    RandomSearchStrategy,
    SequentialGalleryStrategy,
    SimpleBOStrategy,
    TRCBStrategy,
)
from core.pca_utils import LatentProjector


@dataclass
class ProposalBatch:
    latents: np.ndarray
    features: np.ndarray
    backend: str
    roles: list[str] | None = None


class MaximumVarianceLineQueryStrategy:
    name = "mvlq"

    def __init__(
        self,
        generator: FaceGenerator,
        projector: LatentProjector,
        parameters: dict | None = None,
    ) -> None:
        values = parameters or {}
        self.generator = generator
        self.projector = projector
        self.active_dimensions = min(
            int(values.get("latent_dim", projector.dimensions)),
            projector.dimensions,
        )
        self.jitter = 1e-6

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
        del count, seed, round_id
        if not state_path:
            raise ValueError("MVLQ requires a persistent state path")
        center_latent = np.asarray(center, dtype=np.float32)
        center_feature = self.projector.transform(center_latent[None, :])[0]
        state = self._load_or_initialize_state(state_path, center_feature)
        posterior_map = np.asarray(state["map"], dtype=np.float64)
        covariance = np.asarray(state["covariance"], dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (covariance + covariance.T)
        )
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0:
            direction = -direction

        query_scale = max(float(sigma), 0.0)
        radius = float(state["radius0"]) * query_scale
        coefficients = np.linspace(-1.0, 1.0, int(display_count), dtype=np.float64)
        active_queries = (
            posterior_map[None, :]
            + radius * coefficients[:, None] * direction[None, :]
        )
        query_features = np.repeat(
            center_feature[None, :],
            int(display_count),
            axis=0,
        ).astype(np.float32)
        query_features[:, : self.active_dimensions] = active_queries.astype(
            np.float32
        )
        center_reconstruction = self.projector.inverse_transform(
            center_feature[None, :]
        )[0]
        query_latents = (
            center_latent[None, :]
            + self.projector.inverse_transform(query_features)
            - center_reconstruction[None, :]
        ).astype(np.float32)
        return ProposalBatch(
            latents=query_latents,
            features=self.projector.transform(query_latents),
            backend=self.name,
            roles=[
                f"mvlq_line_b={coefficient:.6f}_scale={query_scale:.6f}"
                for coefficient in coefficients
            ],
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
    ) -> tuple[np.ndarray, float]:
        del sigma
        if not state_path:
            raise ValueError("MVLQ requires a persistent state path")
        state = self._load_state(state_path)
        shown_features = self.projector.transform(shown_latents)[
            :, : self.active_dimensions
        ].astype(np.float64)
        winner_feature = self.projector.transform(
            np.atleast_2d(winner_latent)
        )[0, : self.active_dimensions].astype(np.float64)
        winner_index = int(
            np.argmin(np.linalg.norm(shown_features - winner_feature, axis=1))
        )
        history = list(state["history"])
        history.append(
            {
                "queries": shown_features.astype(np.float32),
                "winner_index": winner_index,
            }
        )
        posterior_map, covariance = self._laplace_posterior(
            prior_mean=np.asarray(state["prior_mean"], dtype=np.float64),
            prior_covariance=np.asarray(
                state["prior_covariance"], dtype=np.float64
            ),
            history=history,
            initial=np.asarray(state["map"], dtype=np.float64),
        )
        state["history"] = history
        state["map"] = posterior_map.astype(np.float32)
        state["covariance"] = covariance.astype(np.float32)
        self._save_state(state_path, state)

        center_latent = np.asarray(center, dtype=np.float32)
        center_feature = self.projector.transform(center_latent[None, :])[0]
        next_feature = center_feature.copy()
        next_feature[: self.active_dimensions] = posterior_map.astype(np.float32)
        center_reconstruction = self.projector.inverse_transform(
            center_feature[None, :]
        )[0]
        next_center = (
            center_latent
            + self.projector.inverse_transform(next_feature[None, :])[0]
            - center_reconstruction
        ).astype(np.float32)
        return next_center, 1.0

    def constrain_map(self, feasible_latent: np.ndarray, state_path: str) -> None:
        state = self._load_state(state_path)
        feasible_feature = self.projector.transform(
            np.atleast_2d(feasible_latent)
        )[0, : self.active_dimensions]
        state["map"] = feasible_feature.astype(np.float32)
        state["constraint_projections"] = int(
            state.get("constraint_projections", 0)
        ) + 1
        self._save_state(state_path, state)

    def _load_or_initialize_state(
        self,
        state_path: str,
        center_feature: np.ndarray,
    ) -> dict:
        path = Path(state_path)
        if path.exists():
            return self._load_state(state_path)
        prior_mean = np.asarray(
            center_feature[: self.active_dimensions], dtype=np.float32
        )
        prior_covariance = np.eye(self.active_dimensions, dtype=np.float32)
        state = {
            "algorithm": self.name,
            "version": 1,
            "active_dimensions": self.active_dimensions,
            "prior_mean": prior_mean,
            "prior_covariance": prior_covariance,
            "radius0": float(
                np.sqrt(np.linalg.eigvalsh(prior_covariance).max())
            ),
            "map": prior_mean.copy(),
            "covariance": prior_covariance.copy(),
            "history": [],
            "constraint_projections": 0,
        }
        self._save_state(state_path, state)
        return state

    def _load_state(self, state_path: str) -> dict:
        with Path(state_path).open("rb") as handle:
            state = pickle.load(handle)
        if (
            state.get("algorithm") != self.name
            or int(state.get("active_dimensions", -1)) != self.active_dimensions
        ):
            raise RuntimeError("Stored strategy state is not compatible with MVLQ")
        return state

    @staticmethod
    def _save_state(state_path: str, state: dict) -> None:
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(state, handle)

    def _laplace_posterior(
        self,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray,
        history: list[dict],
        initial: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(len(prior_mean), dtype=np.float64)
        prior_precision = np.linalg.inv(
            prior_covariance + self.jitter * identity
        )
        estimate = np.asarray(initial, dtype=np.float64).copy()
        for _ in range(50):
            gradient, precision = self._gradient_and_precision(
                estimate,
                prior_mean,
                prior_precision,
                history,
            )
            step = np.linalg.solve(
                precision + self.jitter * identity,
                gradient,
            )
            if float(np.linalg.norm(step)) < 1e-7:
                break
            current_objective = self._log_posterior(
                estimate,
                prior_mean,
                prior_precision,
                history,
            )
            step_scale = 1.0
            while step_scale >= 1e-4:
                candidate = estimate + step_scale * step
                if self._log_posterior(
                    candidate,
                    prior_mean,
                    prior_precision,
                    history,
                ) >= current_objective:
                    estimate = candidate
                    break
                step_scale *= 0.5
            else:
                break
        _, precision = self._gradient_and_precision(
            estimate,
            prior_mean,
            prior_precision,
            history,
        )
        covariance = np.linalg.inv(precision + self.jitter * identity)
        return estimate, 0.5 * (covariance + covariance.T)

    @staticmethod
    def _choice_probabilities(queries: np.ndarray, estimate: np.ndarray) -> np.ndarray:
        logits = queries @ estimate - 0.5 * np.sum(queries * queries, axis=1)
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        return probabilities / float(np.sum(probabilities))

    def _gradient_and_precision(
        self,
        estimate: np.ndarray,
        prior_mean: np.ndarray,
        prior_precision: np.ndarray,
        history: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        gradient = -prior_precision @ (estimate - prior_mean)
        precision = prior_precision.copy()
        for observation in history:
            queries = np.asarray(observation["queries"], dtype=np.float64)
            winner_index = int(observation["winner_index"])
            probabilities = self._choice_probabilities(queries, estimate)
            expected_query = probabilities @ queries
            gradient += queries[winner_index] - expected_query
            centered = queries - expected_query[None, :]
            precision += centered.T @ (probabilities[:, None] * centered)
        return gradient, precision

    def _log_posterior(
        self,
        estimate: np.ndarray,
        prior_mean: np.ndarray,
        prior_precision: np.ndarray,
        history: list[dict],
    ) -> float:
        delta = estimate - prior_mean
        value = -0.5 * float(delta @ prior_precision @ delta)
        for observation in history:
            queries = np.asarray(observation["queries"], dtype=np.float64)
            winner_index = int(observation["winner_index"])
            logits = queries @ estimate - 0.5 * np.sum(
                queries * queries, axis=1
            )
            maximum = float(np.max(logits))
            value += float(logits[winner_index] - maximum)
            value -= float(np.log(np.exp(logits - maximum).sum()))
        return value


class DeepIECQueryStrategy:
    name = "deepiec_mutation"

    def __init__(
        self,
        generator: FaceGenerator,
        projector: LatentProjector,
        mutation_probability: float,
        random_immigration_probability: float,
        mutation_decay: float,
        minimum_mutation_scale: float,
        duplicate_distance: float,
        active_dimensions: int | None = None,
    ) -> None:
        self.generator = generator
        self.projector = projector
        self.mutation_probability = mutation_probability
        self.random_immigration_probability = random_immigration_probability
        self.mutation_decay = mutation_decay
        self.minimum_mutation_scale = minimum_mutation_scale
        self.duplicate_distance = duplicate_distance
        self.active_dimensions = min(
            active_dimensions or projector.dimensions,
            projector.dimensions,
        )

    def initial_population(self, count: int, seed: int) -> ProposalBatch:
        latents = self.generator.sample_prior(count, seed)
        return ProposalBatch(
            latents=latents,
            features=self.projector.transform(latents),
            backend=self.name,
            roles=["initial_prior"] * count,
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
        random = np.random.default_rng(seed)
        elite = np.asarray(center, dtype=np.float32)
        elite_full_feature = self.projector.transform(elite[None, :])[0]
        elite_feature = elite_full_feature[: self.active_dimensions]
        elite_reconstruction = self.projector.inverse_transform(
            elite_full_feature[None, :]
        )[0]
        latents = [elite]
        features = [elite_feature]
        roles = ["elite"]
        attempts = 0
        maximum_attempts = max(200, count * 50)

        while len(latents) < count and attempts < maximum_attempts:
            attempts += 1
            if random.random() < self.random_immigration_probability:
                candidate = self.generator.sample_prior(
                    1,
                    seed + attempts * 104729,
                )[0]
                role = "random_immigrant"
            else:
                candidate = elite.copy()
                role = "elite_clone"
                if random.random() < self.mutation_probability:
                    noise = random.normal(
                        0.0,
                        sigma,
                        size=elite_feature.shape,
                    ).astype(np.float32)
                    mutated_feature = elite_full_feature.copy()
                    mutated_feature[: self.active_dimensions] = elite_feature + noise
                    mutated_reconstruction = self.projector.inverse_transform(
                        mutated_feature[None, :]
                    )[0]
                    candidate = (
                        elite + mutated_reconstruction - elite_reconstruction
                    ).astype(np.float32)
                    role = "gaussian_mutation"

            candidate_feature = self.projector.transform(candidate[None, :])[0][
                : self.active_dimensions
            ]
            minimum_distance = min(
                float(np.linalg.norm(candidate_feature - existing))
                for existing in features
            )
            if minimum_distance < self.duplicate_distance:
                continue
            latents.append(candidate)
            features.append(candidate_feature)
            roles.append(role)

        if len(latents) < count:
            fallback = self.generator.sample_prior(
                count - len(latents),
                seed + 999983,
            )
            for candidate in fallback:
                latents.append(candidate)
                features.append(
                    self.projector.transform(candidate[None, :])[0][
                        : self.active_dimensions
                    ]
                )
                roles.append("fallback_immigrant")

        return ProposalBatch(
            latents=np.asarray(latents, dtype=np.float32),
            features=np.asarray(features, dtype=np.float32),
            backend=self.name,
            roles=roles,
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
    ) -> tuple[np.ndarray, float]:
        return (
            np.asarray(winner_latent, dtype=np.float32),
            float(max(self.minimum_mutation_scale, self.mutation_decay * sigma)),
        )


class HeuristicQueryStrategy:
    name = "heuristic"

    def __init__(
        self,
        generator: FaceGenerator,
        projector: LatentProjector,
        alpha: float,
        decay: float,
        min_sigma: float,
        exploration_fraction: float,
        exploration_scale: float,
        max_sigma: float | None = None,
        sigma_down: float | None = None,
        sigma_up: float | None = None,
    ) -> None:
        self.generator = generator
        self.projector = projector
        self.alpha = alpha
        self.decay = decay
        self.min_sigma = min_sigma
        self.exploration_fraction = exploration_fraction
        self.exploration_scale = exploration_scale
        self.max_sigma = max_sigma
        self.sigma_down = sigma_down
        self.sigma_up = sigma_up

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
        random = np.random.default_rng(seed)
        center_latent = np.asarray(center, dtype=np.float32)
        center_feature = self.projector.transform(center_latent[None, :])[0]
        center_reconstruction = self.projector.inverse_transform(
            center_feature[None, :]
        )[0]
        latents = [center_latent]
        features = [center_feature]
        for index in range(1, count):
            if random.random() < self.exploration_fraction:
                candidate = self.generator.sample_prior(
                    1,
                    seed + index * 104729,
                )[0]
            else:
                mutated_feature = center_feature + random.normal(
                    0.0,
                    sigma,
                    size=center_feature.shape,
                ).astype(np.float32)
                candidate = (
                    center_latent
                    + self.projector.inverse_transform(mutated_feature[None, :])[0]
                    - center_reconstruction
                ).astype(np.float32)
            latents.append(candidate)
            features.append(self.projector.transform(candidate[None, :])[0])
        return ProposalBatch(
            latents=np.asarray(latents, dtype=np.float32),
            features=np.asarray(features, dtype=np.float32),
            backend=self.name,
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
    ) -> tuple[np.ndarray, float]:
        selected_incumbent = np.allclose(center, winner_latent)
        center_feature = self.projector.transform(center[None, :])[0]
        winner_feature = self.projector.transform(winner_latent[None, :])[0]
        next_feature = (
            (1.0 - self.alpha) * center_feature + self.alpha * winner_feature
        )
        center_reconstruction = self.projector.inverse_transform(
            center_feature[None, :]
        )[0]
        new_center = (
            center
            + self.projector.inverse_transform(next_feature[None, :])[0]
            - center_reconstruction
        )
        if selected_incumbent and self.sigma_down is not None:
            new_sigma = max(self.min_sigma, self.sigma_down * sigma)
        elif not selected_incumbent and self.sigma_up is not None:
            new_sigma = min(self.max_sigma or np.inf, self.sigma_up * sigma)
        else:
            new_sigma = max(self.min_sigma, self.decay * sigma)
        return new_center.astype(np.float32), float(new_sigma)


class CMAESQueryStrategy:
    name = "cmaes"

    def __init__(
        self,
        generator: FaceGenerator,
        projector: LatentProjector,
        min_sigma: float,
    ) -> None:
        self.generator = generator
        self.projector = projector
        self.min_sigma = min_sigma
        try:
            import cma
        except ImportError as exc:
            raise RuntimeError(
                "CMA-ES mode requires the 'cma' package from requirements.txt"
            ) from exc
        self.cma = cma

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
        if not state_path:
            raise ValueError("CMA-ES requires a persistent state path")
        state_file = Path(state_path)
        if state_file.exists():
            with state_file.open("rb") as handle:
                state = pickle.load(handle)
            strategy = state["strategy"]
        else:
            mean = self.projector.transform(np.atleast_2d(center))[0]
            strategy = self.cma.CMAEvolutionStrategy(
                mean.tolist(),
                sigma,
                {
                    "seed": int(seed),
                    "popsize": max(4, int(count)),
                    "verbose": -9,
                },
            )

        population = np.asarray(strategy.ask(), dtype=np.float32)
        prior_count = max(256, len(population) * 16)
        prior_latents = self.generator.sample_prior(prior_count, seed + 7919)
        prior_features = self.projector.transform(prior_latents)
        available = set(range(prior_count))
        projected_indices: list[int] = []
        for target in population:
            nearest = min(
                available,
                key=lambda index: float(
                    np.linalg.norm(prior_features[index] - target)
                ),
            )
            projected_indices.append(nearest)
            available.remove(nearest)
        projected_latents = prior_latents[projected_indices]
        projected_features = prior_features[projected_indices]
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("wb") as handle:
            pickle.dump({"strategy": strategy, "population": population}, handle)
        return ProposalBatch(
            latents=projected_latents,
            features=projected_features,
            backend=self.name,
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
    ) -> tuple[np.ndarray, float]:
        if not state_path:
            raise ValueError("CMA-ES requires a persistent state path")
        state_file = Path(state_path)
        with state_file.open("rb") as handle:
            state = pickle.load(handle)
        strategy = state["strategy"]
        population = state["population"]

        shown_features = self.projector.transform(shown_latents)
        winner_feature = self.projector.transform(np.atleast_2d(winner_latent))[0]
        losses = np.full(len(population), 1.0, dtype=np.float64)
        for feature in shown_features:
            index = int(np.argmin(np.linalg.norm(population - feature, axis=1)))
            losses[index] = 0.6
        winner_index = int(
            np.argmin(np.linalg.norm(population - winner_feature, axis=1))
        )
        losses[winner_index] = 0.0
        strategy.tell(population.tolist(), losses.tolist())

        with state_file.open("wb") as handle:
            pickle.dump({"strategy": strategy}, handle)
        new_center = self.projector.inverse_transform(
            np.asarray(strategy.mean, dtype=np.float32)[None, :]
        )[0]
        return new_center, float(max(self.min_sigma, strategy.sigma))


def create_query_strategy(
    config,
    generator,
    projector,
    mode: str | None = None,
    parameters: dict | None = None,
):
    selected_mode = mode or config.search.mode
    values = parameters or {}
    if selected_mode == "mvlq":
        return MaximumVarianceLineQueryStrategy(generator, projector, values)
    if selected_mode == "random":
        return RandomSearchStrategy(generator, projector, values)
    if selected_mode == "axis_pair":
        return AxisPairSearchStrategy(generator, projector, values)
    if selected_mode == "fast_direct":
        return FastDirectLatentESStrategy(generator, projector, values)
    if selected_mode == "linear_ucb":
        return LinearPreferenceUCBStrategy(generator, projector, values)
    if selected_mode == "simplebo":
        return SimpleBOStrategy(generator, projector, values)
    if selected_mode == "banditbo":
        return BanditBOStrategy(generator, projector, values)
    if selected_mode == "qeubo":
        return QEUBOStrategy(generator, projector, values)
    if selected_mode == "sequential_gallery":
        return SequentialGalleryStrategy(generator, projector, values)
    if selected_mode == "trcb":
        return TRCBStrategy(generator, projector, values)
    if selected_mode == "deepiec_mutation":
        return DeepIECQueryStrategy(
            generator=generator,
            projector=projector,
            mutation_probability=float(
                values.get(
                    "mutation_probability",
                    config.search.mutation_probability,
                )
            ),
            random_immigration_probability=(
                float(
                    values.get(
                        "foreign_fraction",
                        config.search.random_immigration_probability,
                    )
                )
            ),
            mutation_decay=float(
                values.get("sigma_decay", config.search.mutation_decay)
            ),
            minimum_mutation_scale=float(
                values.get("sigma_min", config.search.minimum_mutation_scale)
            ),
            duplicate_distance=float(
                values.get("duplicate_distance", config.search.duplicate_distance)
            ),
            active_dimensions=int(values.get("latent_dim", projector.dimensions)),
        )
    if selected_mode == "cmaes":
        return CMAESQueryStrategy(
            generator=generator,
            projector=projector,
            min_sigma=float(values.get("sigma_min", config.search.min_sigma)),
        )
    if selected_mode != "heuristic":
        raise ValueError(f"Unknown search strategy: {selected_mode}")
    return HeuristicQueryStrategy(
        generator=generator,
        projector=projector,
        alpha=float(values.get("alpha", config.search.alpha)),
        decay=config.search.decay,
        min_sigma=float(values.get("sigma_min", config.search.min_sigma)),
        exploration_fraction=float(
            values.get(
                "global_fraction",
                config.search.exploration_fraction,
            )
        ),
        exploration_scale=config.search.exploration_scale,
        max_sigma=float(values.get("sigma_max", 1.0)),
        sigma_down=float(values.get("sigma_down", config.search.decay)),
        sigma_up=float(values.get("sigma_up", 1.0)),
    )
