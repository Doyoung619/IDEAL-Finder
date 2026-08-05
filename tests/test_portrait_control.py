from __future__ import annotations

import numpy as np
from PIL import Image

from core.portrait_control import PortraitQualityController


class FakeCLIPRanker:
    def __init__(self, probabilities: np.ndarray | None) -> None:
        self.probabilities = probabilities

    def classify(self, images, prompts):
        return self.probabilities


def test_clean_portrait_must_win_and_clear_threshold():
    images = [Image.new("RGB", (16, 16)) for _ in range(3)]
    controller = PortraitQualityController(
        clip_ranker=FakeCLIPRanker(
            np.asarray(
                [
                    [0.80, 0.10, 0.05, 0.05],
                    [0.40, 0.45, 0.10, 0.05],
                    [0.45, 0.20, 0.20, 0.15],
                ],
                dtype=np.float32,
            )
        ),
        confidence_threshold=0.50,
    )

    estimates = controller.estimate(images, generator_name="stylegan2_ffhq")

    assert controller.accepts(estimates[0])
    assert not controller.accepts(estimates[1])
    assert not controller.accepts(estimates[2])


def test_demo_portraits_bypass_clip():
    controller = PortraitQualityController(
        clip_ranker=FakeCLIPRanker(None),
        confidence_threshold=0.50,
    )

    estimate = controller.estimate(
        [Image.new("RGB", (16, 16))],
        generator_name="demo_torch_decoder",
    )[0]

    assert controller.accepts(estimate)
    assert estimate.backend == "demo_clean_portrait"
