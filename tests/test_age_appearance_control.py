from __future__ import annotations

import numpy as np
from PIL import Image

from core.age_appearance_control import AgeAppearanceController


class FakeCLIPRanker:
    def classify(self, images, prompts):
        return np.asarray(
            [
                [0.70, 0.20, 0.08, 0.02],
                [0.25, 0.60, 0.14, 0.01],
            ],
            dtype=np.float32,
        )


def test_age_appearance_controller_separates_20s_and_30s():
    controller = AgeAppearanceController(FakeCLIPRanker())
    images = [Image.new("RGB", (8, 8)) for _ in range(2)]

    estimates = controller.estimate(images, "stylegan2_ffhq")

    assert estimates[0].label == "20s"
    assert estimates[0].twenties_confidence == np.float32(0.70)
    assert estimates[1].label == "30s"
    assert estimates[1].twenties_thirties_confidence == np.float32(0.85)


def test_demo_age_defaults_to_20s_adult():
    controller = AgeAppearanceController(FakeCLIPRanker())

    estimate = controller.estimate(
        [Image.new("RGB", (8, 8))],
        "demo_torch_portrait_decoder",
    )[0]

    assert estimate.label == "20s"
    assert estimate.twenties_confidence == 1.0
