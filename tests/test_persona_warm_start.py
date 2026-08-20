import pickle

import numpy as np
import pytest

from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.generator import DemoFaceGenerator
from core.query_strategy import EntropyQueryStrategy


def make_strategy(required=True, adaptive_beta_enabled=True):
    prior = ConditionalPCAPrior(
        condition=DemographicCondition("female", ("east_asian",)),
        mu_w=np.zeros(4),
        components=np.eye(4)[:2],
        eigenvalues=np.asarray([4.0, 1.0]),
        explained_variance_ratio=np.asarray([0.8, 0.2]),
        accepted_samples=100,
        seed=1,
    )
    return EntropyQueryStrategy(
        DemoFaceGenerator(latent_dim=4, output_resolution=16, device="cpu"),
        prior,
        {
            "posterior_mc_samples": 8,
            "num_restarts": 1,
            "optimization_steps": 1,
            "learning_rate": 0.05,
            "seed": 1,
            "device": "cpu",
            "warm_start_required": required,
            "adaptive_beta_enabled": adaptive_beta_enabled,
        },
    )


def test_persona_state_is_exact_local_global_mixture(tmp_path):
    strategy = make_strategy()
    theta = np.asarray([0.75, -1.25], dtype=np.float32)
    path = tmp_path / "state.pkl"
    strategy.initialize_state(path, theta, covariance_scale=1.5, metadata={"pool_id": "x"})

    with path.open("rb") as handle:
        state = pickle.load(handle)
    assert np.array_equal(state["mixture"]["local_mean"], theta)
    assert state["mixture"]["global_weight"] == 0.45
    assert np.isclose(state["component_is_global"].mean(), 0.45, atol=1e-4)
    assert np.allclose(state["prior_mean"], 0.55 * theta)
    assert state["particles"].shape == (8192, 2)
    assert np.isclose(state["weights"].sum(), 1.0)
    assert state["history"] == []
    assert state["initialization"]["type"] == "persona_warm_start"


def test_required_strategy_never_silently_initializes_at_demographic_mean(tmp_path):
    strategy = make_strategy(required=True)
    with pytest.raises(RuntimeError, match="Persona warm start is required"):
        strategy.propose(
            center=np.zeros(4),
            sigma=1.0,
            count=5,
            seed=1,
            state_path=str(tmp_path / "missing.pkl"),
            display_count=5,
        )


def test_recommended_fixed_beta_mode_ignores_ratings_deterministically():
    strategy = make_strategy(adaptive_beta_enabled=False)
    first = strategy._adapt_beta(1.0, preference_rating=10, difficulty_rating=1)
    second = strategy._adapt_beta(1.0, preference_rating=1, difficulty_rating=7)
    assert first == 1.0
    assert second == 1.0
