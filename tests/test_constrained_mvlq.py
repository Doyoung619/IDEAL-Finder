import numpy as np
import pytest

from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.constrained_mvlq import (
    DemographicConstrainedMVLQ,
    NoFeasibleQuerySetError,
)
from core.mock_models import MockDemographicClassifier, MockStyleGANGenerator
from core.mvlq import line_coefficients, maximum_variance_line_queries


def make_prior(mu=(1.0, 1.0, 0.0)) -> ConditionalPCAPrior:
    return ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.array(mu, dtype=float),
        components=np.eye(3)[:2],
        eigenvalues=np.ones(2),
        explained_variance_ratio=np.array([0.5, 0.5]),
        accepted_samples=20,
    )


def test_mvlq_queries_are_uniform_on_one_line():
    queries, coefficients, direction, _ = maximum_variance_line_queries(
        np.zeros(2), np.diag([2.0, 1.0]), count=5, radius=2.0
    )
    offsets = queries - queries[2]
    orthogonal = offsets - (offsets @ direction)[:, None] * direction[None, :]

    assert np.allclose(coefficients, line_coefficients(5))
    assert np.allclose(np.diff(coefficients), 0.5)
    assert np.allclose(orthogonal, 0.0)


def test_shared_alpha_backtracks_and_all_queries_are_feasible():
    prior = make_prior()
    strategy = DemographicConstrainedMVLQ(
        prior,
        MockStyleGANGenerator(w_dimension=3),
        MockDemographicClassifier(),
        gender_threshold=0.62,
        race_threshold=0.85,
        center_candidates=0,
    )
    result = strategy.propose(
        posterior_map=np.zeros(2),
        posterior_mean=np.zeros(2),
        posterior_covariance=np.diag([2.0, 1.0]),
        count=5,
        radius=2.0,
    )

    projected = (result.theta_queries - result.center) @ result.direction
    nonzero = np.abs(result.coefficients) > 0
    inferred_alpha = projected[nonzero] / (
        2.0 * result.coefficients[nonzero]
    )
    assert result.common_alpha < 1.0
    assert result.backtracking_steps > 0
    assert np.allclose(inferred_alpha, result.common_alpha)
    assert np.all(result.gender_probabilities >= 0.62)
    assert np.all(result.race_probabilities >= 0.85)


def test_invalid_map_uses_prior_center_fallback():
    prior = make_prior()
    strategy = DemographicConstrainedMVLQ(
        prior,
        MockStyleGANGenerator(w_dimension=3),
        MockDemographicClassifier(),
        gender_threshold=0.7,
        race_threshold=0.7,
        center_candidates=0,
    )
    result = strategy.propose(
        posterior_map=np.array([-3.0, 0.0]),
        posterior_mean=np.array([-3.0, 0.0]),
        posterior_covariance=np.diag([0.1, 0.05]),
        count=3,
        radius=0.5,
    )

    assert result.fallback_used
    assert result.center_source == "prior_mean"


def test_no_feasible_set_raises_explicit_error():
    prior = make_prior(mu=(-3.0, -3.0, 0.0))
    strategy = DemographicConstrainedMVLQ(
        prior,
        MockStyleGANGenerator(w_dimension=3),
        MockDemographicClassifier(),
        gender_threshold=0.999,
        race_threshold=0.999,
        center_candidates=0,
    )
    with pytest.raises(NoFeasibleQuerySetError) as captured:
        strategy.propose(
            posterior_map=np.zeros(2),
            posterior_covariance=np.eye(2) * 0.01,
            count=3,
            radius=0.2,
        )

    assert captured.value.details["condition"]["gender"] == "female"
    assert captured.value.details["centers"]
