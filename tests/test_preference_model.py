from __future__ import annotations

import numpy as np

from core.pca_utils import LatentProjector
from core.preference_model import PairwisePreferenceModel


def test_pairwise_preference_model_ranks_winners_higher(tmp_path):
    random = np.random.default_rng(5)
    latent_bank = random.normal(size=(200, 10)).astype(np.float32)
    projector = LatentProjector(tmp_path / "pca.joblib", dimensions=6).fit(
        latent_bank
    )
    direction = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    pairs = []
    for _ in range(30):
        base = random.normal(size=10).astype(np.float32)
        pairs.append((base + direction, base - direction))

    model = PairwisePreferenceModel(projector).fit(pairs)
    winners = np.vstack([winner for winner, _ in pairs])
    losers = np.vstack([loser for _, loser in pairs])

    assert model.fitted
    assert float(model.score(winners).mean()) > float(model.score(losers).mean())

    path = tmp_path / "preference.joblib"
    model.save(path)
    loaded = PairwisePreferenceModel(projector).load(path)
    assert np.allclose(model.score(winners), loaded.score(winners))

