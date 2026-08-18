import numpy as np

from core.preference_posterior import ParticleMixturePreferencePosterior
from core.rc_mlq import RCMLQConfig, RCMLQSelector


def posterior(covariance, seed=3):
    return ParticleMixturePreferencePosterior.from_gaussian_mixture(
        local_mean=np.zeros(3),
        local_covariance=np.asarray(covariance),
        global_mean=np.zeros(3),
        global_covariance=np.asarray(covariance),
        global_weight=0.45,
        particle_count=4096,
        seed=seed,
        beta=1.0,
    )


def selector():
    return RCMLQSelector(
        RCMLQConfig(
            resolution_min=0.2,
            resolution_max=6.0,
            resolution_steps=40,
            posterior_samples=1024,
        )
    )


def test_direction_solves_maximum_posterior_variance_argmax():
    result = selector().select(posterior(np.diag([0.2, 2.0, 0.5])), 8, seed=4)
    assert result.query_points.shape == (8, 3)
    assert abs(float(result.direction[1])) > 0.999
    assert np.isclose(result.principal_variance, 2.0, rtol=0.12)
    assert result.min_pairwise_distance > 0


def test_resolution_is_selected_from_fixed_physical_grid_not_alpha_sigma():
    wide = selector().select(posterior(np.eye(3), seed=9), 4, seed=5)
    narrow = selector().select(posterior(np.eye(3) * 0.01, seed=9), 4, seed=5)
    assert 0.2 <= wide.physical_resolution <= 6.0
    assert 0.2 <= narrow.physical_resolution <= 6.0
    assert narrow.relative_resolution > narrow.physical_resolution
    assert np.isfinite(narrow.mutual_information)
