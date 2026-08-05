from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from core.generator import FaceGenerator
from core.pca_utils import LatentProjector


class StrategySupport:
    name = "strategy"

    def __init__(
        self,
        generator: FaceGenerator,
        projector: LatentProjector,
        parameters: dict,
    ) -> None:
        self.generator = generator
        self.projector = projector
        self.parameters = parameters
        self.active_dimensions = min(
            int(parameters.get("latent_dim", projector.dimensions)),
            projector.dimensions,
        )

    def features(self, latents: np.ndarray) -> np.ndarray:
        return self.projector.transform(latents)[:, : self.active_dimensions]

    def replace_features(
        self,
        reference: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        reference_latent = np.asarray(reference, dtype=np.float32)
        reference_full = self.projector.transform(reference_latent[None, :])[0]
        target_full = np.repeat(
            reference_full[None, :],
            len(np.atleast_2d(features)),
            axis=0,
        )
        target_full[:, : self.active_dimensions] = np.atleast_2d(features)
        reference_reconstruction = self.projector.inverse_transform(
            reference_full[None, :]
        )[0]
        target_reconstruction = self.projector.inverse_transform(target_full)
        return (
            reference_latent[None, :]
            + target_reconstruction
            - reference_reconstruction[None, :]
        ).astype(np.float32)

    def prior(self, count: int, seed: int):
        latents = self.generator.sample_prior(count, seed)
        return latents, self.features(latents)

    @staticmethod
    def load_state(state_path: str | None, default: dict) -> dict:
        if state_path and Path(state_path).exists():
            with Path(state_path).open("rb") as handle:
                return pickle.load(handle)
        return default

    @staticmethod
    def save_state(state_path: str | None, state: dict) -> None:
        if not state_path:
            return
        destination = Path(state_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(state, handle)


class RandomSearchStrategy(StrategySupport):
    name = "random"

    def initial_population(self, count: int, seed: int, **_kwargs):
        from core.query_strategy import ProposalBatch

        latents, features = self.prior(count, seed)
        return ProposalBatch(
            latents=latents,
            features=features,
            backend=self.name,
            roles=["random_prior"] * count,
        )

    def propose(self, count: int, seed: int, **_kwargs):
        return self.initial_population(count=count, seed=seed)

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        return np.asarray(center, dtype=np.float32), float(sigma)


class AxisPairSearchStrategy(StrategySupport):
    name = "axis_pair"

    def propose(
        self,
        center: np.ndarray,
        sigma: float,
        count: int,
        seed: int,
        round_id: int = 1,
        display_count: int = 2,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        center_feature = self.features(center[None, :])[0]
        include_incumbent = bool(self.parameters.get("include_incumbent", False))
        pair_slots = display_count - 1 if include_incumbent else display_count
        axis_count = max(1, pair_slots // 2)
        start_axis = ((round_id - 1) * axis_count) % self.active_dimensions
        feature_candidates: list[np.ndarray] = []
        roles: list[str] = []
        if include_incumbent:
            feature_candidates.append(center_feature.copy())
            roles.append("incumbent")
        for offset in range(axis_count):
            axis = (start_axis + offset) % self.active_dimensions
            for sign in (1.0, -1.0):
                candidate = center_feature.copy()
                candidate[axis] += sign * sigma
                feature_candidates.append(candidate)
                roles.append(f"axis_{axis}_{'plus' if sign > 0 else 'minus'}")
        if len(feature_candidates) < display_count:
            feature_candidates.append(center_feature.copy())
            roles.append("incumbent")

        random = np.random.default_rng(seed)
        while len(feature_candidates) < count:
            axis = int(random.integers(0, self.active_dimensions))
            candidate = center_feature.copy()
            candidate[axis] += float(random.choice([-1.0, 1.0])) * sigma * float(
                random.uniform(0.45, 1.6)
            )
            feature_candidates.append(candidate)
            roles.append(f"axis_{axis}_backup")
        features = np.asarray(feature_candidates[:count], dtype=np.float32)
        latents = self.replace_features(center, features)
        return ProposalBatch(latents, features, self.name, roles[:count])

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        winner_latent: np.ndarray,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        center_feature = self.features(center[None, :])[0]
        winner_feature = self.features(winner_latent[None, :])[0]
        move_rate = float(self.parameters["move_rate"])
        next_feature = center_feature + move_rate * (
            winner_feature - center_feature
        )
        next_center = self.replace_features(center, next_feature[None, :])[0]
        next_sigma = max(
            float(self.parameters["step_min"]),
            float(self.parameters["step_decay"]) * sigma,
        )
        return next_center, float(next_sigma)


class FastDirectLatentESStrategy(StrategySupport):
    name = "fast_direct"

    def propose(
        self,
        center: np.ndarray,
        sigma: float,
        count: int,
        seed: int,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        random = np.random.default_rng(seed)
        center_feature = self.features(center[None, :])[0]
        features = [center_feature.copy()]
        roles = ["incumbent"]
        direction = None
        sign = 1.0
        while len(features) < count:
            if direction is None or sign > 0:
                direction = random.standard_normal(
                    self.active_dimensions
                ).astype(np.float32)
                direction /= np.linalg.norm(direction) + 1e-8
                sign = -1.0
            else:
                sign = 1.0
            scale = sigma * float(random.choice([0.55, 0.85, 1.0, 1.25]))
            features.append(center_feature + sign * scale * direction)
            roles.append("antithetic_gaussian")
        feature_array = np.asarray(features, dtype=np.float32)
        return ProposalBatch(
            latents=self.replace_features(center, feature_array),
            features=feature_array,
            backend=self.name,
            roles=roles,
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        winner_latent: np.ndarray,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        selected_incumbent = np.allclose(center, winner_latent)
        if selected_incumbent:
            next_sigma = max(
                float(self.parameters["sigma_min"]),
                sigma * float(self.parameters["sigma_shrink"]),
            )
        else:
            next_sigma = min(
                float(self.parameters["sigma_max"]),
                sigma * float(self.parameters["sigma_expand"]),
            )
        return np.asarray(winner_latent, dtype=np.float32), float(next_sigma)


class LinearPreferenceUCBStrategy(StrategySupport):
    name = "linear_ucb"

    def propose(
        self,
        center: np.ndarray,
        count: int,
        seed: int,
        state_path: str | None = None,
        display_count: int = 2,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        latents, features = self.prior(count, seed)
        state = self.load_state(state_path, {"pairs": []})
        theta, precision = self._fit(state["pairs"])
        inverse_precision = np.linalg.pinv(precision)
        uncertainty = np.sqrt(
            np.maximum(
                np.einsum("ni,ij,nj->n", features, inverse_precision, features),
                0.0,
            )
        )
        scores = features @ theta + float(self.parameters["ucb_beta"]) * uncertainty
        order = diversity_order(
            features,
            scores,
            float(self.parameters["diversity_lambda"]),
            float(self.parameters["diversity_lengthscale"]),
            preferred_count=display_count,
        )
        return ProposalBatch(
            latents=latents[order],
            features=features[order],
            backend=self.name,
            roles=["linear_ucb"] * len(order),
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        state = self.load_state(state_path, {"pairs": []})
        winner = self.features(winner_latent[None, :])[0]
        shown = self.features(shown_latents)
        weight = 1.0 / max(len(shown) - 1, 1)
        for loser in shown:
            if not np.allclose(loser, winner):
                state["pairs"].append((winner.copy(), loser.copy(), weight))
        self.save_state(state_path, state)
        return np.asarray(winner_latent, dtype=np.float32), float(sigma)

    def _fit(self, pairs: list) -> tuple[np.ndarray, np.ndarray]:
        ridge = float(self.parameters["ridge_lambda"])
        if not pairs:
            return (
                np.zeros(self.active_dimensions, dtype=np.float32),
                ridge * np.eye(self.active_dimensions, dtype=np.float32),
            )
        differences = np.asarray(
            [winner - loser for winner, loser, _ in pairs],
            dtype=np.float64,
        )
        weights = np.asarray([weight for _, _, weight in pairs], dtype=np.float64)
        x = np.vstack([differences, -differences])
        y = np.concatenate([np.ones(len(pairs)), np.zeros(len(pairs))])
        sample_weights = np.concatenate([weights, weights])
        model = LogisticRegression(
            C=1.0 / ridge,
            fit_intercept=False,
            max_iter=1000,
            random_state=0,
        )
        model.fit(x, y, sample_weight=sample_weights)
        theta = model.coef_[0]
        probabilities = sigmoid(differences @ theta)
        curvature = weights * probabilities * (1.0 - probabilities)
        precision = ridge * np.eye(self.active_dimensions)
        precision += differences.T @ (curvature[:, None] * differences)
        return theta.astype(np.float32), precision.astype(np.float32)


class SimpleBOStrategy(StrategySupport):
    name = "simplebo"

    def propose(
        self,
        count: int,
        seed: int,
        state_path: str | None = None,
        display_count: int = 2,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        latents, features = self.prior(count, seed)
        state = self.load_state(state_path, {"pairs": []})
        posterior = PreferenceGP(
            state["pairs"],
            lengthscale=float(self.parameters["kernel_lengthscale"]),
            preference_noise=float(self.parameters["preference_noise"]),
        )
        mean, variance = posterior.predict(features)
        scores = mean + float(self.parameters["ucb_beta"]) * np.sqrt(variance)
        order = diversity_order(
            features,
            scores,
            float(self.parameters["diversity_lambda"]),
            float(self.parameters["diversity_lengthscale"]),
            preferred_count=display_count,
        )
        return ProposalBatch(
            latents[order],
            features[order],
            self.name,
            ["gp_ucb"] * len(order),
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        append_pairwise_state(self, state_path, shown_latents, winner_latent)
        return np.asarray(winner_latent, dtype=np.float32), float(sigma)


class QEUBOStrategy(SimpleBOStrategy):
    name = "qeubo"

    def propose(
        self,
        count: int,
        seed: int,
        state_path: str | None = None,
        display_count: int = 2,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        latents, features = self.prior(count, seed)
        state = self.load_state(state_path, {"pairs": []})
        posterior = PreferenceGP(
            state["pairs"],
            lengthscale=float(self.parameters["kernel_lengthscale"]),
            preference_noise=float(self.parameters["preference_noise"]),
        )
        mean, covariance = posterior.joint(features)
        random = np.random.default_rng(seed + 1299709)
        sample_count = int(self.parameters["posterior_samples"])
        jitter = 1e-5 * np.eye(len(features))
        try:
            factor = np.linalg.cholesky(covariance + jitter)
            samples = mean[:, None] + factor @ random.standard_normal(
                (len(features), sample_count)
            )
        except np.linalg.LinAlgError:
            samples = mean[:, None] + np.sqrt(
                np.maximum(np.diag(covariance), 1e-6)
            )[:, None] * random.standard_normal((len(features), sample_count))

        selected = [int(np.argmax(mean))]
        best_samples = samples[selected[0]].copy()
        while len(selected) < min(display_count, len(features)):
            gains = np.mean(
                np.maximum(samples, best_samples[None, :])
                - best_samples[None, :],
                axis=1,
            )
            gains[selected] = -np.inf
            next_index = int(np.argmax(gains))
            selected.append(next_index)
            best_samples = np.maximum(best_samples, samples[next_index])
        remaining = [
            int(index)
            for index in np.argsort(mean)[::-1]
            if int(index) not in selected
        ]
        order = np.asarray(selected + remaining, dtype=int)
        return ProposalBatch(
            latents[order],
            features[order],
            self.name,
            ["qeubo_joint"] * len(order),
        )


class BanditBOStrategy(StrategySupport):
    name = "banditbo"

    def propose(
        self,
        center: np.ndarray,
        count: int,
        seed: int,
        round_id: int = 1,
        state_path: str | None = None,
        display_count: int = 2,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        initial_count = int(self.parameters["initial_count"])
        initial_reward = float(self.parameters["initial_mean_reward"])
        state = self.load_state(
            state_path,
            {
                "counts": np.full(self.active_dimensions, initial_count, dtype=float),
                "mean_rewards": np.full(
                    self.active_dimensions,
                    initial_reward,
                    dtype=float,
                ),
                "axis_pairs": [[] for _ in range(self.active_dimensions)],
            },
        )
        counts = np.asarray(state["counts"], dtype=float)
        rewards = np.asarray(state["mean_rewards"], dtype=float)
        bonuses = np.sqrt(
            float(self.parameters["bandit_alpha"])
            * np.log(max(round_id, 2))
            / np.maximum(counts, 1.0)
        )
        dimension = int(np.argmax(rewards + bonuses))
        center_feature = self.features(center[None, :])[0]
        grid = np.linspace(
            float(self.parameters["coordinate_min"]),
            float(self.parameters["coordinate_max"]),
            int(self.parameters["grid_size"]),
            dtype=np.float32,
        )
        one_dimensional_pairs = [
            (
                np.asarray([winner], dtype=np.float32),
                np.asarray([loser], dtype=np.float32),
                weight,
            )
            for winner, loser, weight in state["axis_pairs"][dimension]
        ]
        posterior = PreferenceGP(
            one_dimensional_pairs,
            lengthscale=1.0,
            preference_noise=0.25,
        )
        mean, variance = posterior.predict(grid[:, None])
        scores = mean + float(self.parameters["ucb_beta"]) * np.sqrt(variance)
        order = diversity_order(
            grid[:, None],
            scores,
            float(self.parameters["diversity_lambda"]),
            0.5,
            preferred_count=display_count,
        )
        incumbent_coordinate = float(center_feature[dimension])
        coordinate_values = [incumbent_coordinate]
        coordinate_values.extend(
            float(grid[index])
            for index in order
            if abs(float(grid[index]) - incumbent_coordinate) > 1e-5
        )
        coordinate_values = coordinate_values[:count]
        features = np.repeat(center_feature[None, :], len(coordinate_values), axis=0)
        features[:, dimension] = coordinate_values
        latents = self.replace_features(center, features)
        state["selected_dimension"] = dimension
        self.save_state(state_path, state)
        roles = ["incumbent"] + [f"axis_{dimension}_bo"] * (len(features) - 1)
        return ProposalBatch(latents, features, self.name, roles)

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        state = self.load_state(state_path, {})
        dimension = int(state["selected_dimension"])
        counts = np.asarray(state["counts"], dtype=float)
        rewards = np.asarray(state["mean_rewards"], dtype=float)
        center_feature = self.features(center[None, :])[0]
        winner_feature = self.features(winner_latent[None, :])[0]
        reward = float(
            abs(float(winner_feature[dimension] - center_feature[dimension])) > 1e-5
        )
        counts[dimension] += 1.0
        rewards[dimension] += (reward - rewards[dimension]) / counts[dimension]
        state["counts"] = counts
        state["mean_rewards"] = rewards
        shown = self.features(shown_latents)
        weight = 1.0 / max(len(shown) - 1, 1)
        for loser in shown:
            if not np.allclose(loser, winner_feature):
                state["axis_pairs"][dimension].append(
                    (
                        float(winner_feature[dimension]),
                        float(loser[dimension]),
                        weight,
                    )
                )
        self.save_state(state_path, state)
        return np.asarray(winner_latent, dtype=np.float32), float(sigma)


class SequentialGalleryStrategy(StrategySupport):
    name = "sequential_gallery"

    def propose(
        self,
        center: np.ndarray,
        sigma: float,
        count: int,
        seed: int,
        display_count: int = 2,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        random = np.random.default_rng(seed)
        directions = random.standard_normal((2, self.active_dimensions))
        directions[0] /= np.linalg.norm(directions[0]) + 1e-8
        directions[1] -= directions[0] * np.dot(directions[0], directions[1])
        directions[1] /= np.linalg.norm(directions[1]) + 1e-8
        rows, columns = gallery_shape(display_count)
        row_values = np.linspace(-sigma, sigma, rows)
        column_values = np.linspace(-sigma, sigma, columns)
        center_feature = self.features(center[None, :])[0]
        features = [
            center_feature + row * directions[0] + column * directions[1]
            for row in row_values
            for column in column_values
        ]
        roles = ["gallery_grid"] * len(features)
        while len(features) < count:
            offsets = random.uniform(-sigma, sigma, size=2)
            features.append(
                center_feature
                + offsets[0] * directions[0]
                + offsets[1] * directions[1]
            )
            roles.append("gallery_backup")
        feature_array = np.asarray(features[:count], dtype=np.float32)
        return ProposalBatch(
            self.replace_features(center, feature_array),
            feature_array,
            self.name,
            roles[:count],
        )

    def update(
        self,
        sigma: float,
        winner_latent: np.ndarray,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        next_sigma = max(
            float(self.parameters["step_min"]),
            float(self.parameters["step_decay"]) * sigma,
        )
        return np.asarray(winner_latent, dtype=np.float32), float(next_sigma)


class TRCBStrategy(StrategySupport):
    name = "trcb"

    def propose(
        self,
        count: int,
        seed: int,
        state_path: str | None = None,
        **_kwargs,
    ):
        from core.query_strategy import ProposalBatch

        bank_size = max(count, int(self.parameters["bank_size"]))
        state = self.load_state(state_path, {})
        if "bank_latents" not in state:
            bank_latents, bank_features = self.prior(bank_size, seed + 1009)
            state = {
                "bank_latents": bank_latents,
                "bank_features": bank_features,
                "wins": np.zeros((bank_size, bank_size), dtype=np.float32),
            }
        wins = np.asarray(state["wins"], dtype=np.float64)
        smoothing = float(self.parameters["smoothing"])
        comparisons = wins + wins.T
        empirical = (wins + smoothing) / (comparisons + 2.0 * smoothing)
        base_scores = empirical.mean(axis=1)
        uncertainty = np.sqrt(
            np.log(max(float(comparisons.sum()), 2.0))
            / (comparisons.sum(axis=1) + 1.0)
        )
        random = np.random.default_rng(seed)
        optimistic = base_scores + float(
            self.parameters["confidence_scale"]
        ) * uncertainty * random.uniform(-1.0, 1.0, size=len(base_scores))
        utilities = np.maximum(optimistic, 1e-4) ** float(self.parameters["gamma"])
        order = np.argsort(utilities)[::-1]
        state["last_order"] = order
        self.save_state(state_path, state)
        selected = order[:count]
        return ProposalBatch(
            np.asarray(state["bank_latents"])[selected],
            np.asarray(state["bank_features"])[selected],
            self.name,
            ["fixed_bank"] * len(selected),
        )

    def update(
        self,
        center: np.ndarray,
        sigma: float,
        shown_latents: np.ndarray,
        winner_latent: np.ndarray,
        state_path: str | None = None,
        **_kwargs,
    ) -> tuple[np.ndarray, float]:
        state = self.load_state(state_path, {})
        bank = np.asarray(state["bank_latents"])
        wins = np.asarray(state["wins"])
        winner_index = nearest_row(bank, winner_latent)
        for loser in shown_latents:
            loser_index = nearest_row(bank, loser)
            if loser_index != winner_index:
                wins[winner_index, loser_index] += 1.0
        state["wins"] = wins
        self.save_state(state_path, state)
        return np.asarray(winner_latent, dtype=np.float32), float(sigma)


class PreferenceGP:
    def __init__(
        self,
        pairs: list,
        lengthscale: float,
        preference_noise: float,
    ) -> None:
        self.lengthscale = lengthscale
        self.preference_noise = preference_noise
        self.points = np.empty((0, 1), dtype=np.float64)
        self.mode = np.empty(0, dtype=np.float64)
        self.kernel_inverse = np.empty((0, 0), dtype=np.float64)
        self.posterior_covariance = np.empty((0, 0), dtype=np.float64)
        if pairs:
            self._fit(pairs)

    def _fit(self, pairs: list) -> None:
        point_list: list[np.ndarray] = []
        indices: dict[bytes, int] = {}

        def point_index(point: np.ndarray) -> int:
            key = np.asarray(point, dtype=np.float32).tobytes()
            if key not in indices:
                indices[key] = len(point_list)
                point_list.append(np.asarray(point, dtype=np.float64))
            return indices[key]

        comparisons = []
        for winner, loser, weight in pairs:
            comparisons.append(
                (point_index(winner), point_index(loser), float(weight))
            )
        self.points = np.vstack(point_list)
        kernel = matern52(self.points, self.points, self.lengthscale)
        kernel += 1e-4 * np.eye(len(self.points))
        self.kernel_inverse = np.linalg.pinv(kernel)
        mode = np.zeros(len(self.points), dtype=np.float64)
        hessian = self.kernel_inverse.copy()
        for _ in range(25):
            gradient = -self.kernel_inverse @ mode
            hessian = self.kernel_inverse.copy()
            for winner, loser, weight in comparisons:
                difference = (
                    mode[winner] - mode[loser]
                ) / self.preference_noise
                probability = float(sigmoid(np.asarray([difference]))[0])
                coefficient = weight * (1.0 - probability) / self.preference_noise
                gradient[winner] += coefficient
                gradient[loser] -= coefficient
                curvature = (
                    weight
                    * probability
                    * (1.0 - probability)
                    / (self.preference_noise**2)
                )
                hessian[winner, winner] += curvature
                hessian[loser, loser] += curvature
                hessian[winner, loser] -= curvature
                hessian[loser, winner] -= curvature
            step = np.linalg.solve(hessian + 1e-8 * np.eye(len(mode)), gradient)
            mode += step
            if np.linalg.norm(step) < 1e-6:
                break
        self.mode = mode
        self.posterior_covariance = np.linalg.pinv(hessian)

    def predict(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.mode):
            return (
                np.zeros(len(candidates), dtype=np.float32),
                np.ones(len(candidates), dtype=np.float32),
            )
        cross = matern52(candidates, self.points, self.lengthscale)
        mean = cross @ self.kernel_inverse @ self.mode
        correction = (
            self.kernel_inverse
            - self.kernel_inverse
            @ self.posterior_covariance
            @ self.kernel_inverse
        )
        variance = 1.0 - np.einsum(
            "ni,ij,nj->n",
            cross,
            correction,
            cross,
        )
        return mean.astype(np.float32), np.maximum(variance, 1e-6).astype(np.float32)

    def joint(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prior = matern52(candidates, candidates, self.lengthscale)
        if not len(self.mode):
            return np.zeros(len(candidates), dtype=np.float32), prior
        cross = matern52(candidates, self.points, self.lengthscale)
        mean = cross @ self.kernel_inverse @ self.mode
        correction = (
            self.kernel_inverse
            - self.kernel_inverse
            @ self.posterior_covariance
            @ self.kernel_inverse
        )
        covariance = prior - cross @ correction @ cross.T
        covariance = (covariance + covariance.T) / 2.0
        return mean.astype(np.float32), covariance.astype(np.float64)


def append_pairwise_state(
    strategy: StrategySupport,
    state_path: str | None,
    shown_latents: np.ndarray,
    winner_latent: np.ndarray,
) -> None:
    state = strategy.load_state(state_path, {"pairs": []})
    winner = strategy.features(winner_latent[None, :])[0]
    shown = strategy.features(shown_latents)
    weight = 1.0 / max(len(shown) - 1, 1)
    for loser in shown:
        if not np.allclose(loser, winner):
            state["pairs"].append((winner.copy(), loser.copy(), weight))
    strategy.save_state(state_path, state)


def diversity_order(
    features: np.ndarray,
    scores: np.ndarray,
    diversity_lambda: float,
    lengthscale: float,
    preferred_count: int | None = None,
) -> np.ndarray:
    if not len(features):
        return np.asarray([], dtype=int)
    selected = [int(np.argmax(scores))]
    remaining = set(range(len(features))) - set(selected)
    target = min(preferred_count or len(features), len(features))
    while remaining and len(selected) < target:
        best_index = None
        best_value = -np.inf
        for index in remaining:
            distances = np.linalg.norm(features[index] - features[selected], axis=1)
            similarity = float(
                np.max(np.exp(-(distances**2) / (2.0 * lengthscale**2)))
            )
            value = float(scores[index]) - diversity_lambda * similarity
            if value > best_value:
                best_value = value
                best_index = index
        selected.append(int(best_index))
        remaining.remove(int(best_index))
    selected.extend(
        int(index)
        for index in np.argsort(scores)[::-1]
        if int(index) in remaining
    )
    return np.asarray(selected, dtype=int)


def matern52(
    left: np.ndarray,
    right: np.ndarray,
    lengthscale: float,
) -> np.ndarray:
    distances = np.linalg.norm(
        np.atleast_2d(left)[:, None, :] - np.atleast_2d(right)[None, :, :],
        axis=2,
    )
    scaled = np.sqrt(5.0) * distances / max(lengthscale, 1e-6)
    return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def gallery_shape(display_count: int) -> tuple[int, int]:
    return {
        2: (1, 2),
        4: (2, 2),
        8: (2, 4),
        16: (4, 4),
    }.get(display_count, (1, display_count))


def nearest_row(values: np.ndarray, target: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(values - target[None, :], axis=1)))
