from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class EntropyQueryConfig:
    """Numerical options for M-way mutual-information query synthesis."""

    posterior_mc_samples: int = 512
    num_restarts: int = 8
    optimization_steps: int = 200
    learning_rate: float = 0.05
    seed: int = 0
    device: str = "cpu"
    output_spread_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.posterior_mc_samples < 2:
            raise ValueError("posterior_mc_samples must be at least two")
        if self.num_restarts < 1 or self.optimization_steps < 1:
            raise ValueError("restart and optimization step counts must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.output_spread_scale < 1.0:
            raise ValueError("output_spread_scale must be at least one")


@dataclass
class EntropyQueryResult:
    """Optimized latent query points and mutual-information diagnostics."""

    query_points: np.ndarray
    mutual_information: float
    predictive_entropy: float
    expected_conditional_entropy: float
    min_pairwise_distance: float
    max_mahalanobis_radius: float
    restart: int
    optimization_steps: int
    initial_mutual_information: float
    spread_scale: float = 1.0
    pre_spread_mutual_information: float | None = None
    pre_spread_min_pairwise_distance: float | None = None

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable optimization metrics."""
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
            "spread_scale": self.spread_scale,
            "pre_spread_mutual_information": self.pre_spread_mutual_information,
            "pre_spread_min_pairwise_distance": self.pre_spread_min_pairwise_distance,
        }


def choice_probabilities(
    posterior_samples: torch.Tensor,
    query_points: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the stable distance-based M-way softmax choice model."""
    if posterior_samples.ndim != 2 or query_points.ndim != 2:
        raise ValueError("posterior_samples and query_points must be matrices")
    if posterior_samples.shape[1] != query_points.shape[1]:
        raise ValueError("posterior samples and query points must share a dimension")
    squared_distances = torch.sum(
        (posterior_samples[:, None, :] - query_points[None, :, :]) ** 2,
        dim=-1,
    )
    logits = -float(beta) * squared_distances
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return probabilities, log_probabilities


def mutual_information(
    posterior_samples: torch.Tensor,
    query_points: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute predictive entropy minus expected conditional entropy."""
    probabilities, log_probabilities = choice_probabilities(
        posterior_samples, query_points, beta=beta
    )
    epsilon = torch.finfo(probabilities.dtype).eps
    predictive = probabilities.mean(dim=0)
    predictive_entropy = -torch.sum(
        predictive * torch.log(predictive.clamp_min(epsilon))
    )
    conditional_entropies = -torch.sum(
        probabilities * log_probabilities, dim=-1
    )
    expected_conditional_entropy = conditional_entropies.mean()
    score = predictive_entropy - expected_conditional_entropy
    return score, predictive_entropy, expected_conditional_entropy


class EntropyQuerySelector:
    """Synthesize M latent queries by directly maximizing response information."""

    def __init__(self, config: EntropyQueryConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

    def select(
        self,
        posterior,
        prior_covariance: np.ndarray,
        num_options: int,
        seed: int | None = None,
    ) -> EntropyQueryResult:
        """Optimize queries inside the MAP-centered prior ellipsoid."""
        if num_options < 2:
            raise ValueError("num_options must be at least two")
        current_seed = self.config.seed if seed is None else int(seed)
        dtype = torch.float32
        center = torch.as_tensor(
            self._value(posterior, "map_estimate"),
            dtype=dtype,
            device=self.device,
        )
        dimension = int(center.numel())
        if center.ndim != 1:
            raise ValueError("posterior MAP must be one-dimensional")
        prior_factor = self._stable_cholesky(
            prior_covariance, dtype=dtype, device=self.device
        )
        if prior_factor.shape != (dimension, dimension):
            raise ValueError("prior covariance dimension does not match the posterior")
        samples = posterior.sample(
            self.config.posterior_mc_samples,
            seed=current_seed,
            device=self.device,
            dtype=dtype,
        )
        if samples.shape != (self.config.posterior_mc_samples, dimension):
            raise ValueError("posterior.sample returned an invalid shape")

        random = torch.Generator(device=self.device)
        random.manual_seed(current_seed + 104729)
        best: dict[str, Any] | None = None
        failures: list[str] = []
        for restart in range(self.config.num_restarts):
            raw_u = torch.nn.Parameter(
                0.1
                * torch.randn(
                    (num_options, dimension),
                    generator=random,
                    dtype=dtype,
                    device=self.device,
                )
            )
            optimizer = torch.optim.Adam(
                [raw_u], lr=self.config.learning_rate
            )
            with torch.no_grad():
                self._project_unit_ball(raw_u)
                initial_queries = center[None, :] + raw_u @ prior_factor.T
                initial_score, _, _ = mutual_information(
                    samples, initial_queries, beta=float(posterior.beta)
                )
            if not torch.isfinite(initial_score):
                failures.append(f"restart {restart}: non-finite initial score")
                continue
            local_best = self._snapshot(
                raw_u,
                samples,
                center,
                prior_factor,
                restart,
                float(initial_score.item()),
                beta=float(posterior.beta),
            )
            failed = False
            for _ in range(self.config.optimization_steps):
                optimizer.zero_grad(set_to_none=True)
                queries = center[None, :] + raw_u @ prior_factor.T
                score, _, _ = mutual_information(
                    samples, queries, beta=float(posterior.beta)
                )
                if not torch.isfinite(score):
                    failures.append(f"restart {restart}: non-finite objective")
                    failed = True
                    break
                (-score).backward()
                if raw_u.grad is None or not torch.isfinite(raw_u.grad).all():
                    failures.append(f"restart {restart}: non-finite gradient")
                    failed = True
                    break
                optimizer.step()
                with torch.no_grad():
                    self._project_unit_ball(raw_u)
                    candidate = self._snapshot(
                        raw_u,
                        samples,
                        center,
                        prior_factor,
                        restart,
                        float(initial_score.item()),
                        beta=float(posterior.beta),
                    )
                if candidate["mutual_information"] > local_best["mutual_information"]:
                    local_best = candidate
            if failed:
                continue
            if best is None or local_best["mutual_information"] > best["mutual_information"]:
                best = local_best
        if best is None:
            raise RuntimeError(
                "All entropy-query optimization restarts failed: "
                + "; ".join(failures)
            )
        if not np.isfinite(best["query_points"]).all() or not np.isfinite(
            best["mutual_information"]
        ):
            raise RuntimeError("Entropy-query optimization returned non-finite values")
        pre_spread_information = float(best["mutual_information"])
        pre_spread_distance = float(best["min_pairwise_distance"])
        spread_scale = float(self.config.output_spread_scale)
        if spread_scale > 1.0:
            expanded = center[None, :] + spread_scale * (
                torch.as_tensor(best["query_points"], device=self.device) - center[None, :]
            )
            score, predictive, conditional = mutual_information(
                samples, expanded, beta=float(posterior.beta)
            )
            best["query_points"] = expanded.detach().cpu().numpy().astype(np.float32)
            best["mutual_information"] = float(score.item())
            best["predictive_entropy"] = float(predictive.item())
            best["expected_conditional_entropy"] = float(conditional.item())
            best["min_pairwise_distance"] = float(torch.pdist(expanded).min().item())
            best["max_mahalanobis_radius"] *= spread_scale
        return EntropyQueryResult(
            query_points=best["query_points"],
            mutual_information=best["mutual_information"],
            predictive_entropy=best["predictive_entropy"],
            expected_conditional_entropy=best["expected_conditional_entropy"],
            min_pairwise_distance=best["min_pairwise_distance"],
            max_mahalanobis_radius=best["max_mahalanobis_radius"],
            restart=best["restart"],
            optimization_steps=self.config.optimization_steps,
            initial_mutual_information=best["initial_mutual_information"],
            spread_scale=spread_scale,
            pre_spread_mutual_information=pre_spread_information,
            pre_spread_min_pairwise_distance=pre_spread_distance,
        )

    @staticmethod
    def _project_unit_ball(values: torch.Tensor) -> None:
        norms = values.norm(dim=-1, keepdim=True)
        values.div_(torch.clamp(norms, min=1.0))

    @staticmethod
    def _value(posterior, name: str):
        value = getattr(posterior, name)
        return value() if callable(value) else value

    @staticmethod
    def _stable_cholesky(
        covariance: np.ndarray,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        matrix = torch.as_tensor(covariance, dtype=dtype, device=device)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("covariance must be square")
        if not torch.isfinite(matrix).all():
            raise ValueError("covariance contains NaN or Inf")
        matrix = 0.5 * (matrix + matrix.T)
        identity = torch.eye(matrix.shape[0], dtype=dtype, device=device)
        for jitter in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            factor, info = torch.linalg.cholesky_ex(matrix + jitter * identity)
            if int(info.max().item()) == 0:
                return factor
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        repaired = eigenvectors @ torch.diag(eigenvalues.clamp_min(1e-8)) @ eigenvectors.T
        factor, info = torch.linalg.cholesky_ex(repaired + 1e-6 * identity)
        if int(info.max().item()) != 0:
            raise RuntimeError("Unable to stabilize covariance for entropy query")
        return factor

    @staticmethod
    def _snapshot(
        raw_u: torch.Tensor,
        samples: torch.Tensor,
        center: torch.Tensor,
        prior_factor: torch.Tensor,
        restart: int,
        initial_score: float,
        beta: float,
    ) -> dict[str, Any]:
        queries = center[None, :] + raw_u @ prior_factor.T
        score, predictive, conditional = mutual_information(
            samples, queries, beta=beta
        )
        if len(queries) > 1:
            distances = torch.pdist(queries)
            minimum_distance = float(distances.min().item())
        else:
            minimum_distance = 0.0
        return {
            "query_points": queries.detach().cpu().numpy().astype(np.float32),
            "mutual_information": float(score.item()),
            "predictive_entropy": float(predictive.item()),
            "expected_conditional_entropy": float(conditional.item()),
            "min_pairwise_distance": minimum_distance,
            "max_mahalanobis_radius": float(
                raw_u.norm(dim=-1).max().item()
            ),
            "restart": restart,
            "initial_mutual_information": initial_score,
        }
