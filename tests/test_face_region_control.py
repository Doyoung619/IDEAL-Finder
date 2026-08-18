from __future__ import annotations

import numpy as np
from PIL import Image
from core.face_region_control import FaceRegionController


class FakeCLIPRanker:
    def classify(self, images, prompts):
        return np.asarray(
            [
                [0.75, 0.10, 0.05, 0.03, 0.04, 0.03],
                [0.10, 0.70, 0.05, 0.05, 0.05, 0.05],
            ],
            dtype=np.float32,
        )


def test_face_region_controller_exposes_east_asian_score():
    controller = FaceRegionController(FakeCLIPRanker())
    images = [Image.new("RGB", (8, 8)) for _ in range(2)]

    estimates = controller.estimate(images, "stylegan2_ffhq")

    assert estimates[0].label == "east_asian"
    assert estimates[0].east_asian_confidence == 0.75
    assert estimates[1].label == "white_european"


def test_demo_region_bypasses_clip():
    controller = FaceRegionController(FakeCLIPRanker())

    estimate = controller.estimate(
        [Image.new("RGB", (8, 8))],
        "demo_torch_portrait_decoder",
    )[0]

    assert estimate.label == "east_asian"
    assert estimate.east_asian_confidence == 1.0
