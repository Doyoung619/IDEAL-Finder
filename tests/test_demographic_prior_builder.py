import numpy as np

from core.conditional_prior import DemographicCondition
from core.demographic_prior_builder import (
    DemographicPriorBuilder,
    PriorBuildConfig,
    hard_selection_mask,
    soft_selection_weights,
)
from core.mock_models import MockDemographicClassifier, MockStyleGANGenerator


def test_hard_filter_and_soft_weights():
    gender = np.array([0.95, 0.89, 0.91])
    race = np.array([0.81, 0.99, 0.70])
    assert hard_selection_mask(gender, race, 0.9, 0.8).tolist() == [
        True,
        False,
        False,
    ]
    assert np.allclose(soft_selection_weights(gender, race), gender * race)


def test_builder_supports_hard_and_soft_modes(tmp_path):
    generator = MockStyleGANGenerator(w_dimension=6)
    classifier = MockDemographicClassifier()
    builder = DemographicPriorBuilder(generator, classifier)
    condition = DemographicCondition("female", ("east_asian",))

    hard = builder.build(
        PriorBuildConfig(
            condition=condition,
            latent_dimension=3,
            selection_mode="hard",
            num_generator_samples=160,
            min_accepted_samples=5,
            batch_size=20,
            gender_threshold=0.7,
            race_threshold=0.7,
            cache_path=str(tmp_path / "bank.npz"),
            seed=3,
        )
    )
    soft = builder.build(
        PriorBuildConfig(
            condition=condition,
            latent_dimension=3,
            selection_mode="soft",
            num_generator_samples=160,
            min_accepted_samples=5,
            cache_path=str(tmp_path / "bank.npz"),
            seed=3,
        )
    )

    assert hard.dimension == soft.dimension == 3
    assert hard.components.shape == (3, 6)
    assert np.all(hard.eigenvalues > 0)
    assert np.allclose(hard.components @ hard.components.T, np.eye(3), atol=1e-5)
