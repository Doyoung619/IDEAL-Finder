import numpy as np

from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.preference_posterior import GaussianPreferencePosterior


def make_prior() -> ConditionalPCAPrior:
    return ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.zeros(4),
        components=np.eye(4)[:2],
        eigenvalues=np.ones(2),
        explained_variance_ratio=np.array([0.5, 0.5]),
        accepted_samples=20,
    )


def test_posterior_initializes_as_standard_normal():
    posterior = GaussianPreferencePosterior.initialize_from_prior(make_prior())
    assert np.array_equal(posterior.mean, np.zeros(2))
    assert np.array_equal(posterior.covariance, np.eye(2))


def test_one_round_preference_update_moves_toward_winner():
    posterior = GaussianPreferencePosterior.initialize_from_prior(make_prior())
    queries = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    before = posterior.choice_probabilities(queries)
    posterior.update(queries, winner_index=2)
    after = posterior.choice_probabilities(queries)

    assert np.allclose(before, [0.27406862, 0.45186276, 0.27406862])
    assert posterior.mean[0] > 0
    assert after[2] > before[2]
    assert np.linalg.eigvalsh(posterior.covariance).min() > 0
