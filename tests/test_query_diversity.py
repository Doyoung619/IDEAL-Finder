import numpy as np
import pytest
from PIL import Image

from core.clip_ranker import CLIPRanker
from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.generator import DemoFaceGenerator
from core.query_diversity import QueryDiversityConfig, ensure_query_diversity


def prior() -> ConditionalPCAPrior:
    return ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.zeros(8),
        components=np.eye(8)[:4],
        eigenvalues=np.ones(4),
        explained_variance_ratio=np.full(4, 0.25),
        accepted_samples=100,
        coordinate_clip=1.5,
    )


def dependencies():
    generator = DemoFaceGenerator(latent_dim=8, output_resolution=64, device="cpu")
    embedder = CLIPRanker(enabled=False, allow_mock=True)
    return generator, embedder


def test_disabled_guard_is_diagnostic_only():
    theta = np.zeros((3, 4), dtype=np.float32)
    generator, embedder = dependencies()
    final, metadata = ensure_query_diversity(
        theta,
        np.diag([1.0, 0.8, 0.4, 0.2]),
        prior(),
        generator,
        embedder,
        QueryDiversityConfig(enabled=False),
    )
    assert np.array_equal(final, theta)
    assert not metadata["diversity_guard_triggered"]
    assert metadata["num_corrected_queries"] == 0
    assert metadata["max_pairwise_image_similarity"] is None


class FailingGenerator:
    def decode(self, latents):
        raise AssertionError("latent-only diversity must not render images")


class FailingEmbedder:
    def encode_images(self, images):
        raise AssertionError("latent-only diversity must not encode images")


def test_latent_guard_expands_queries_without_rendering_images():
    theta = np.array(
        [[-0.1, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    final, metadata = ensure_query_diversity(
        theta,
        np.diag([1.0, 0.8, 0.4, 0.2]),
        prior(),
        FailingGenerator(),
        FailingEmbedder(),
        QueryDiversityConfig(
            enabled=True,
            latent_min_distance=0.75,
            image_similarity_threshold=None,
        ),
    )
    assert metadata["diversity_guard_triggered"]
    assert metadata["pre_guard_min_pairwise_theta_distance"] == pytest.approx(0.2)
    assert metadata["min_pairwise_theta_distance"] == pytest.approx(0.75)
    assert np.linalg.norm(final[0] - final[1]) == pytest.approx(0.75)


def test_guard_moves_only_duplicates_along_bounded_secondary_directions():
    theta = np.zeros((3, 4), dtype=np.float32)
    theta[2, 0] = 0.8
    generator, embedder = dependencies()
    final, metadata = ensure_query_diversity(
        theta,
        np.diag([1.0, 0.8, 0.4, 0.2]),
        prior(),
        generator,
        embedder,
        QueryDiversityConfig(
            enabled=True,
            latent_min_distance=0.15,
            image_similarity_threshold=0.99,
            max_retries=2,
            perturbation_initial=0.10,
            perturbation_max=0.20,
        ),
    )
    assert metadata["diversity_guard_triggered"]
    assert metadata["num_corrected_queries"] == 1
    assert np.array_equal(final[0], theta[0])
    assert np.array_equal(final[2], theta[2])
    assert 0 < np.linalg.norm(final[1] - theta[1]) <= 0.20 + 1e-7
    assert np.max(np.abs(final)) <= 1.5
    assert metadata["original_theta"] == theta.tolist()
    assert np.allclose(metadata["final_theta"], final)


class DummyGenerator:
    def decode(self, latents):
        return [Image.new("RGB", (8, 8), "white") for _ in latents]


class StagedEmbedder:
    def __init__(self):
        self.calls = 0

    def encode_images(self, images):
        self.calls += 1
        if self.calls < 3:
            return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(images), 1))
        return np.eye(len(images), dtype=np.float32)


def test_rc_mlq_expands_primary_radius_before_orthogonal_fallback():
    conditional_prior = ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.zeros(3),
        components=np.eye(3),
        eigenvalues=np.ones(3),
        explained_variance_ratio=np.full(3, 1 / 3),
        accepted_samples=100,
    )
    particles = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -0.5, 0.0],
         [0.0, 0.5, 0.0], [0.0, 0.0, -0.2], [0.0, 0.0, 0.2]]
    )
    weights = np.full(len(particles), 1.0 / len(particles))
    covariance = np.diag([1.0, 0.3, 0.1])
    theta = np.array([[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    final, metadata = ensure_query_diversity(
        theta,
        covariance,
        conditional_prior,
        DummyGenerator(),
        StagedEmbedder(),
        QueryDiversityConfig(
            enabled=True,
            image_similarity_threshold=0.9,
            radius_expansion_factors=(1.1, 1.2, 1.35),
            max_relative_ig_loss=0.05,
            orthogonal_fallback=True,
        ),
        posterior_particles=particles,
        posterior_weights=weights,
        algorithm="rc_mlq",
    )
    assert metadata["radius_expansion_factor"] == 1.2
    assert len(metadata["radius_expansion_candidates"]) == 2
    assert not metadata["orthogonal_fallback_used"]
    assert np.allclose(final[:, 1:], 0.0)
    assert np.ptp(final[:, 0]) > np.ptp(theta[:, 0])
