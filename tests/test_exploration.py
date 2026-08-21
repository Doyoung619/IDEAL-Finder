import numpy as np

from core.exploration import GlobalExplorationPosteriorView, acquisition_posterior


class Posterior:
    mean = np.array([1.0, 0.0])
    map_estimate = mean
    covariance = np.diag([0.1, 0.2])
    prior_covariance = np.eye(2)
    beta = 1.0
    history = []
    particles = np.array([[0.8, 0.0], [1.2, 0.0]])
    weights = np.array([0.5, 0.5])


def test_zero_rho_preserves_original_posterior():
    posterior = Posterior()
    assert acquisition_posterior(posterior, np.zeros(2), np.eye(2), 0.0) is posterior


def test_positive_rho_reopens_global_acquisition_support():
    posterior = Posterior()
    view = acquisition_posterior(
        posterior,
        global_mean=np.zeros(2),
        global_covariance=np.eye(2),
        rho=0.10,
    )
    assert isinstance(view, GlobalExplorationPosteriorView)
    assert np.allclose(view.mean, [0.9, 0.0])
    assert view.covariance[0, 0] > posterior.covariance[0, 0]
    first = np.asarray(view.sample(100, seed=7, device="cpu"))
    second = np.asarray(view.sample(100, seed=7, device="cpu"))
    assert np.array_equal(first, second)
    assert first.shape == (100, 2)
