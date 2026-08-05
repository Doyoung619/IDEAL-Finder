from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from core.clip_ranker import CLIPRanker


@dataclass(frozen=True)
class AdultEstimate:
    is_adult: bool
    confidence: float
    backend: str


class AdultFaceController:
    prompts = (
        "a portrait photograph of a young adult person age 18 to 39",
        "a portrait photograph of a child or teenager under 18 years old",
    )

    def __init__(
        self,
        clip_ranker: CLIPRanker,
        confidence_threshold: float = 0.56,
    ) -> None:
        self.clip_ranker = clip_ranker
        self.confidence_threshold = confidence_threshold

    def estimate(
        self,
        images: Sequence[Image.Image],
        generator_name: str,
    ) -> list[AdultEstimate]:
        if generator_name.startswith("demo"):
            return [
                AdultEstimate(
                    is_adult=True,
                    confidence=1.0,
                    backend="demo_adult_only_decoder",
                )
                for _ in images
            ]
        probabilities = self.clip_ranker.classify(images, self.prompts)
        if probabilities is None:
            return [
                AdultEstimate(
                    is_adult=False,
                    confidence=0.0,
                    backend="unavailable",
                )
                for _ in images
            ]
        return self.from_probabilities(probabilities)

    def from_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> list[AdultEstimate]:
        return [
            AdultEstimate(
                is_adult=float(row[0]) >= float(row[1]),
                confidence=float(row[0]),
                backend="open_clip_zero_shot_shared_image_features",
            )
            for row in probabilities
        ]

    def accepts(self, estimate: AdultEstimate) -> bool:
        return estimate.is_adult and estimate.confidence >= self.confidence_threshold
