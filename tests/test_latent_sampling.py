from __future__ import annotations

import numpy as np

from core.generator import DemoFaceGenerator
from core.latent_sampler import farthest_point_sampling, sample_around_center
from core.pca_utils import LatentProjector


def test_sampling_and_diversity_are_reproducible(tmp_path):
    generator = DemoFaceGenerator(latent_dim=8, output_resolution=32, device="cpu")
    projector = LatentProjector(tmp_path / "pca.joblib", dimensions=4)
    projector.fit(generator.sample_prior(64, seed=10))
    center = generator.sample_prior(1, seed=11)[0]

    first_latents, first_features = sample_around_center(
        center,
        sigma=0.8,
        count=12,
        projector=projector,
        generator=generator,
        seed=12,
    )
    second_latents, second_features = sample_around_center(
        center,
        sigma=0.8,
        count=12,
        projector=projector,
        generator=generator,
        seed=12,
    )

    assert first_latents.shape == (12, 8)
    assert np.allclose(first_latents, second_latents)
    assert np.allclose(first_features, second_features)
    selected = farthest_point_sampling(first_features, count=4, seed=13)
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_demo_faces_are_decoded_from_latents():
    generator = DemoFaceGenerator(latent_dim=8, output_resolution=48, device="cpu")
    latents = generator.sample_prior(2, seed=3)
    images = generator.decode(latents)

    assert [image.size for image in images] == [(48, 48), (48, 48)]
    assert not np.array_equal(np.asarray(images[0]), np.asarray(images[1]))
