from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from core.clip_ranker import CLIPRanker


@dataclass(frozen=True)
class GenderEstimate:
    label: str
    confidence: float
    backend: str


class GenderController:
    prompts = (
        "a studio portrait photograph of a man age 18 or older",
        "a studio portrait photograph of a woman age 18 or older",
    )

    def __init__(
        self,
        clip_ranker: CLIPRanker,
        confidence_threshold: float = 0.58,
        strict: bool = True,
    ) -> None:
        self.clip_ranker = clip_ranker
        self.confidence_threshold = confidence_threshold
        self.strict = strict

    def estimate(
        self,
        images: Sequence[Image.Image],
        latents: np.ndarray | None = None,
        generator_name: str = "",
    ) -> list[GenderEstimate]:
        probabilities = self.clip_ranker.classify(images, self.prompts)
        if probabilities is not None:
            return self.from_probabilities(probabilities)

        if latents is not None and generator_name.startswith("demo"):
            values = np.atleast_2d(latents)[:, 0]
            return [
                GenderEstimate(
                    label="male" if value >= 0 else "female",
                    confidence=float(0.65 + 0.3 * min(abs(value), 1.0)),
                    backend="demo_latent_axis",
                )
                for value in values
            ]
        return [
            GenderEstimate(label="unknown", confidence=0.0, backend="unavailable")
            for _ in images
        ]

    def from_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> list[GenderEstimate]:
        return [
            GenderEstimate(
                label="male" if int(np.argmax(row)) == 0 else "female",
                confidence=float(np.max(row)),
                backend="open_clip_zero_shot_shared_image_features",
            )
            for row in probabilities
        ]

    def accepts(self, estimate: GenderEstimate, target: str | None) -> bool:
        if target in {None, "", "any"}:
            return True
        if estimate.label == "unknown":
            return not self.strict
        return (
            estimate.label == target
            and estimate.confidence >= self.confidence_threshold
        )
