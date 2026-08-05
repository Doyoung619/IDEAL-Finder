import inspect

import numpy as np
import pytest
import torch

from app.settings import load_config
from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.entropy_query import (
    EntropyQueryConfig,
    EntropyQuerySelector,
    mutual_information,
)
from core.mock_models import MockStyleGANGenerator
from core.preference_posterior import GaussianPreferencePosterior
from core.query_strategy import create_query_strategy


def make_prior() -> ConditionalPCAPrior:
    return ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.array([0.8, 0.8, 0.0, 0.0]),
        components=np.eye(4)[:3],
        eigenvalues=np.array([1.5, 0.8, 0.3]),
        explained_variance_ratio=np.array([0.55, 0.3, 0.15]),
        accepted_samples=100,
        theta_covariance=np.diag([1.0, 0.5, 0.25]),
        seed=5,
    )


def make_selector(seed: int = 11) -> EntropyQuerySelector:
    return EntropyQuerySelector(
        EntropyQueryConfig(
            posterior_mc_samples=96,
            num_restarts=2,
            optimization_steps=35,
            learning_rate=0.05,
            seed=seed,
            device="cpu",
        )
    )


def test_shape_finite_and_prior_ellipsoid_constraint():
    prior = make_prior()
    posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    result = make_selector().select(
        posterior, prior.theta_covariance, num_options=5
    )
    delta = result.query_points - posterior.map_estimate[None, :]
    inverse = np.linalg.inv(prior.theta_covariance)
    radii = np.einsum("ni,ij,nj->n", delta, inverse, delta)

    assert result.query_points.shape == (5, 3)
    assert np.isfinite(result.query_points).all()
    assert np.isfinite(result.mutual_information)
    assert np.all(radii <= 1.0 + 1e-5)
    assert result.max_mahalanobis_radius <= 1.0 + 1e-5


def test_exploration_radius_expands_query_ellipsoid():
    prior = make_prior()
    posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    selector = EntropyQuerySelector(
        EntropyQueryConfig(
            posterior_mc_samples=64,
            num_restarts=2,
            optimization_steps=20,
            exploration_radius=1.5,
            device="cpu",
        )
    )
    result = selector.select(posterior, prior.theta_covariance, num_options=4)
    delta = result.query_points - posterior.map_estimate[None, :]
    inverse = np.linalg.inv(prior.theta_covariance)
    radii = np.einsum("ni,ij,nj->n", delta, inverse, delta)

    assert result.exploration_radius == 1.5
    assert np.all(radii <= 2.25 + 1e-5)
    assert result.max_mahalanobis_radius <= 1.5 + 1e-5


def test_duplicate_queries_have_zero_information_and_score_is_nonnegative():
    samples = torch.randn(128, 3)
    duplicates = torch.zeros(5, 3)
    score, predictive, conditional = mutual_information(samples, duplicates)

    assert abs(float(score)) < 1e-6
    assert torch.isfinite(predictive)
    assert torch.isfinite(conditional)

    prior = make_prior()
    posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    optimized = make_selector().select(posterior, prior.theta_covariance, 4)
    assert optimized.mutual_information >= -1e-5


def test_optimization_never_loses_selected_restart_initial_score():
    prior = make_prior()
    posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    result = make_selector().select(posterior, prior.theta_covariance, 4)
    assert (
        result.mutual_information
        >= result.initial_mutual_information - 1e-6
    )


def test_same_seed_reproduces_identical_query_set():
    prior = make_prior()
    first_posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    second_posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    first = make_selector(seed=17).select(
        first_posterior, prior.theta_covariance, 4, seed=23
    )
    second = make_selector(seed=17).select(
        second_posterior, prior.theta_covariance, 4, seed=23
    )
    assert np.array_equal(first.query_points, second.query_points)
    assert first.mutual_information == second.mutual_information


def test_posterior_update_and_end_to_end_decoder_round():
    prior = make_prior()
    posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    before_mean = posterior.mean.copy()
    before_covariance = posterior.covariance.copy()
    result = make_selector().select(posterior, prior.theta_covariance, 5)
    generator = MockStyleGANGenerator(w_dimension=prior.w_dimension)
    images = generator.synthesize_w(prior.theta_to_w(result.query_points))
    true_preference = np.array([0.5, -0.3, 0.1])
    selected = int(
        np.argmax(
            posterior.choice_probabilities(
                result.query_points, estimate=true_preference
            )
        )
    )
    posterior.update(result.query_points, selected)

    assert images.shape == (5, 3, 16, 16)
    assert not np.array_equal(posterior.mean, before_mean)
    assert not np.array_equal(posterior.covariance, before_covariance)
    assert np.isfinite(posterior.map_estimate).all()


def test_factory_and_registry_reject_non_entropy_without_fallback():
    config = load_config(demo_override=True)
    prior = make_prior()
    generator = MockStyleGANGenerator(w_dimension=prior.w_dimension)
    strategy = create_query_strategy(
        config,
        generator,
        mode="entropy",
        conditional_prior=prior,
        parameters={
            "posterior_mc_samples": 32,
            "num_restarts": 1,
            "optimization_steps": 5,
        },
    )
    assert strategy.name == "entropy"
    with pytest.raises(
        ValueError, match="Only the entropy query algorithm is currently supported"
    ):
        create_query_strategy(
            config,
            generator,
            mode="unsupported",
            conditional_prior=prior,
        )
    config.query._values["constraint"] = "unsupported"
    with pytest.raises(ValueError, match="prior_ellipsoid"):
        create_query_strategy(
            config,
            generator,
            mode="entropy",
            conditional_prior=prior,
        )


def test_selector_has_no_decoder_or_candidate_retrieval_dependency():
    signature = inspect.signature(EntropyQuerySelector.select)
    source = inspect.getsource(EntropyQuerySelector.select).lower()
    assert "decoder" not in signature.parameters
    assert "decode" not in source
    assert "nearest" not in source
    assert "topk" not in source
    assert "candidate_pool" not in source
