from __future__ import annotations

import numpy as np

from core.generator import DemoFaceGenerator
from core.pca_utils import LatentProjector
from core.query_strategy import DeepIECQueryStrategy


def test_deepiec_preserves_elite_and_mutates_in_pca_space(tmp_path):
    generator = DemoFaceGenerator(latent_dim=12, output_resolution=32, device="cpu")
    projector = LatentProjector(tmp_path / "pca.joblib", dimensions=6)
    projector.fit(generator.sample_prior(128, seed=1))
    strategy = DeepIECQueryStrategy(
        generator=generator,
        projector=projector,
        mutation_probability=1.0,
        random_immigration_probability=0.0,
        mutation_decay=0.95,
        minimum_mutation_scale=0.1,
        duplicate_distance=0.01,
    )
    elite = generator.sample_prior(1, seed=2)[0]

    proposal = strategy.propose(
        center=elite,
        sigma=0.5,
        count=8,
        seed=3,
    )

    assert np.array_equal(proposal.latents[0], elite)
    assert proposal.roles[0] == "elite"
    assert proposal.roles[1:] == ["gaussian_mutation"] * 7
    assert all(
        np.linalg.norm(proposal.features[index] - proposal.features[0]) >= 0.01
        for index in range(1, 8)
    )

    winner = proposal.latents[3]
    next_center, next_sigma = strategy.update(
        center=elite,
        sigma=0.5,
        shown_latents=proposal.latents,
        winner_latent=winner,
    )
    assert np.array_equal(next_center, winner)
    assert next_sigma == 0.475


def test_deepiec_initial_population_comes_from_prior(tmp_path):
    generator = DemoFaceGenerator(latent_dim=8, output_resolution=32, device="cpu")
    projector = LatentProjector(tmp_path / "pca.joblib", dimensions=4)
    projector.fit(generator.sample_prior(64, seed=10))
    strategy = DeepIECQueryStrategy(
        generator=generator,
        projector=projector,
        mutation_probability=0.5,
        random_immigration_probability=0.1,
        mutation_decay=0.95,
        minimum_mutation_scale=0.1,
        duplicate_distance=0.01,
    )

    population = strategy.initial_population(count=16, seed=11)

    assert population.latents.shape == (16, 8)
    assert population.roles == ["initial_prior"] * 16
    assert np.allclose(population.latents, generator.sample_prior(16, seed=11))

