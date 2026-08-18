import json

import numpy as np

from core.conditional_prior import ConditionalPCAPrior, DemographicCondition


def make_prior() -> ConditionalPCAPrior:
    return ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.arange(6, dtype=np.float64),
        components=np.eye(6, dtype=np.float64)[:3],
        eigenvalues=np.array([4.0, 2.0, 1.0]),
        explained_variance_ratio=np.array([0.5, 0.3, 0.2]),
        accepted_samples=50,
        thresholds={"gender": 0.9, "race": 0.8},
        seed=17,
    )


def test_conditional_prior_round_trip_and_coordinates(tmp_path):
    prior = make_prior()
    theta = np.array([[0.2, -1.0, 2.0], [1.2, 0.5, -0.3]])

    reconstructed = prior.w_to_theta(prior.theta_to_w(theta))
    assert np.allclose(reconstructed, theta, atol=1e-6)
    assert np.allclose(prior.theta_to_w(np.zeros(3)), prior.mu_w)

    path = tmp_path / "prior.npz"
    prior.save(path)
    loaded = ConditionalPCAPrior.load(path)
    assert loaded.condition == prior.condition
    assert np.allclose(loaded.components, prior.components)
    assert np.allclose(loaded.eigenvalues, prior.eigenvalues)
    assert loaded.accepted_samples == 50


def test_prior_shapes_eigenvalues_and_sampling_are_reproducible():
    prior = make_prior()
    first = prior.sample_theta(4)
    second = prior.sample_theta(4)

    assert prior.components.shape == (3, 6)
    assert np.all(prior.eigenvalues > 0)
    assert first.shape == (4, 3)
    assert np.array_equal(first, second)
    assert np.isfinite(prior.log_prob(first)).all()


def test_loads_validated_ideal_stylegan3_prior_schema(tmp_path):
    metadata = {
        "target": "male",
        "candidate_count": 3000,
        "generator": "NVIDIA StyleGAN3-R FFHQ-U 256x256",
        "target_definition": "East Asian, Male, age 20-39",
        "thresholds": {"east_asian": 0.35, "gender": 0.55},
    }
    path = tmp_path / "ideal.npz"
    np.savez_compressed(
        path,
        mu=np.zeros(6, dtype=np.float32),
        basis=np.eye(6, dtype=np.float32)[:, :3],
        eigvals=np.asarray([4.0, 2.0, 1.0], dtype=np.float32),
        latent_dim=np.asarray(3),
        clip=np.asarray(1.5, dtype=np.float32),
        scale=np.asarray(0.65, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    prior = ConditionalPCAPrior.load(path)

    assert prior.condition == DemographicCondition("male", ("east_asian",))
    assert prior.dimension == 3
    assert prior.w_dimension == 6
    assert prior.coordinate_clip == 1.5
    assert np.isclose(prior.sampling_scale, 0.65)
    assert np.max(np.abs(prior.sample_theta(64))) <= 1.5
