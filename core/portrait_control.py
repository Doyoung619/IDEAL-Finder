from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from core.clip_ranker import CLIPRanker


@dataclass(frozen=True)
class PortraitEstimate:
    is_clean_portrait: bool
    confidence: float
    backend: str


class PortraitQualityController:
    prompts = (
        "a clean realistic front-facing portrait photograph of one person age "
        "18 or older "
        "with an unobstructed face",
        "a portrait with a face obscured by a large hat, costume, sunglasses, "
        "hands, or objects",
        "a distorted malformed synthetic face with visual artifacts",
        "a drawing, painting, cartoon, or sculpture of a face",
    )

    def __init__(
        self,
        clip_ranker: CLIPRanker,
        confidence_threshold: float = 0.50,
    ) -> None:
        self.clip_ranker = clip_ranker
        self.confidence_threshold = confidence_threshold

    def estimate(
        self,
        images: Sequence[Image.Image],
        generator_name: str,
    ) -> list[PortraitEstimate]:
        if generator_name.startswith("demo"):
            return [
                PortraitEstimate(
                    is_clean_portrait=True,
                    confidence=1.0,
                    backend="demo_clean_portrait",
                )
                for _ in images
            ]
        probabilities = self.clip_ranker.classify(images, self.prompts)
        if probabilities is None:
            return [
                PortraitEstimate(
                    is_clean_portrait=False,
                    confidence=0.0,
                    backend="unavailable",
                )
                for _ in images
            ]
        return self.from_probabilities(probabilities)

    def from_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> list[PortraitEstimate]:
        return [
            PortraitEstimate(
                is_clean_portrait=int(np.argmax(row)) == 0,
                confidence=float(row[0]),
                backend="open_clip_zero_shot_shared_image_features",
            )
            for row in probabilities
        ]

    def accepts(self, estimate: PortraitEstimate) -> bool:
        return (
            estimate.is_clean_portrait
            and estimate.confidence >= self.confidence_threshold
        )
