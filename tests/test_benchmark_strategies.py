from __future__ import annotations

import pickle

import numpy as np

from app.settings import load_config
from core.generator import DemoFaceGenerator
from core.pca_utils import LatentProjector
from core.query_strategy import create_query_strategy


def test_mvlq_queries_maximum_variance_line_and_updates_posterior(tmp_path):
    config = load_config(demo_override=True)
    generator = DemoFaceGenerator(latent_dim=12, output_resolution=24, device="cpu")
    projector = LatentProjector(tmp_path / "pca.joblib", dimensions=8)
    projector.fit(generator.sample_prior(128, seed=1))
    center = generator.sample_prior(1, seed=2)[0]
    state_path = tmp_path / "mvlq.pkl"
    strategy = create_query_strategy(
        config,
        generator,
        projector,
        mode="mvlq",
        parameters={},
    )

    proposal = strategy.propose(
        center=center,
        sigma=1.0,
        count=64,
        seed=3,
        state_path=str(state_path),
        round_id=1,
        display_count=8,
    )
    active = proposal.features[:, :8]
    differences = np.diff(active, axis=0)

    assert proposal.latents.shape == (8, generator.latent_dim)
    assert np.allclose(differences, differences[0], atol=1e-5)
    assert np.linalg.matrix_rank(active - active.mean(axis=0), tol=1e-5) == 1

    next_center, next_sigma = strategy.update(
        center=center,
        sigma=1.0,
        shown_latents=proposal.latents,
        winner_latent=proposal.latents[-1],
        state_path=str(state_path),
    )
    with state_path.open("rb") as handle:
        state = pickle.load(handle)

    assert next_sigma == 1.0
    assert not np.allclose(next_center, center)
    assert len(state["history"]) == 1
    assert np.linalg.eigvalsh(state["covariance"]).max() <= 1.0 + 1e-6
    assert np.isfinite(state["map"]).all()

    next_proposal = strategy.propose(
        center=next_center,
        sigma=1.0,
        count=8,
        seed=4,
        state_path=str(state_path),
        round_id=2,
        display_count=8,
    )
    assert next_proposal.latents.shape == proposal.latents.shape
    assert np.isfinite(next_proposal.features).all()
