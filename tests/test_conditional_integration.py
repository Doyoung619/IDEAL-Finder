import numpy as np

from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.constrained_mvlq import DemographicConstrainedMVLQ
from core.mock_models import MockDemographicClassifier, MockStyleGANGenerator
from core.preference_posterior import GaussianPreferencePosterior


def test_mock_multiround_conditional_mvlq_smoke():
    prior = ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.array([1.2, 1.2, 0.0, 0.0]),
        components=np.eye(4)[:2],
        eigenvalues=np.array([0.25, 0.25]),
        explained_variance_ratio=np.array([0.5, 0.5]),
        accepted_samples=100,
    )
    posterior = GaussianPreferencePosterior.initialize_from_prior(prior)
    strategy = DemographicConstrainedMVLQ(
        prior,
        MockStyleGANGenerator(w_dimension=4),
        MockDemographicClassifier(),
        gender_threshold=0.7,
        race_threshold=0.7,
        center_candidates=4,
        seed=8,
    )
    true_theta = np.array([0.4, -0.2])
    errors = []
    eigenvalues = []
    for _ in range(4):
        result = strategy.propose(
            posterior.map_estimate,
            posterior.covariance,
            count=5,
            radius=0.8,
            posterior_mean=posterior.mean,
        )
        probabilities = posterior.choice_probabilities(
            result.theta_queries, estimate=true_theta
        )
        posterior.update(result.theta_queries, int(np.argmax(probabilities)))
        errors.append(float(np.linalg.norm(posterior.mean - true_theta)))
        eigenvalues.append(float(np.linalg.eigvalsh(posterior.covariance).max()))

    assert np.isfinite(errors).all()
    assert np.isfinite(eigenvalues).all()
    assert eigenvalues[-1] < eigenvalues[0]
