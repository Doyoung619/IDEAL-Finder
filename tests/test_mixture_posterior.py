import numpy as np

from core.preference_posterior import ParticleMixturePreferencePosterior


def make_posterior(seed=7):
    return ParticleMixturePreferencePosterior.from_gaussian_mixture(
        local_mean=np.array([1.5, -0.5]),
        local_covariance=np.eye(2) * 0.2,
        global_mean=np.zeros(2),
        global_covariance=np.eye(2),
        global_weight=0.45,
        particle_count=4096,
        seed=seed,
        beta=1.0,
    )


def test_initial_particle_prior_has_requested_global_mass_and_moments():
    posterior = make_posterior()
    assert np.isclose(posterior.global_mass, 0.45, atol=1e-12)
    assert np.isclose(posterior.effective_sample_size, 4096, rtol=1e-6)
    assert np.allclose(posterior.prior_mean, 0.55 * np.array([1.5, -0.5]))
    assert np.linalg.eigvalsh(posterior.prior_covariance).min() > 0


def test_particle_update_changes_weights_and_preserves_probability_mass():
    posterior = make_posterior()
    before = posterior.weights.copy()
    posterior.update(np.array([[-1.0, 0.0], [1.0, 0.0]]), winner_index=1)
    assert np.isclose(posterior.weights.sum(), 1.0)
    assert not np.array_equal(before, posterior.weights)
    assert posterior.effective_sample_size < len(posterior.particles)
    assert np.isfinite(posterior.mean).all()
    assert np.isfinite(posterior.covariance).all()


def test_low_ess_triggers_deterministic_regularized_resampling():
    first = make_posterior(seed=11)
    second = make_posterior(seed=11)
    queries = np.array([[-5.0, 0.0], [5.0, 0.0]])
    for item in (first, second):
        item.resample_threshold = 0.99
        item.update(queries, winner_index=1)
        assert item.last_resampled
        assert np.isclose(item.effective_sample_size, len(item.particles))
    assert np.array_equal(first.particles, second.particles)
    assert np.array_equal(first.component_is_global, second.component_is_global)
