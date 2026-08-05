from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from core.clip_ranker import CLIPRanker


@dataclass(frozen=True)
class AgeAppearanceEstimate:
    label: str
    twenties_confidence: float
    twenties_thirties_confidence: float
    backend: str


class AgeAppearanceController:
    prompts = (
        "a realistic portrait photograph of a person age 20 to 29",
        "a realistic portrait photograph of a person age 30 to 39",
        "a realistic portrait photograph of a person age 40 or older",
        "a portrait photograph of a child or teenager under 18 years old",
    )
    labels = ("20s", "30s", "40plus", "minor")

    def __init__(self, clip_ranker: CLIPRanker) -> None:
        self.clip_ranker = clip_ranker

    def estimate(
        self,
        images: Sequence[Image.Image],
        generator_name: str,
    ) -> list[AgeAppearanceEstimate]:
        if generator_name.startswith("demo"):
            return [
                AgeAppearanceEstimate(
                    label="20s",
                    twenties_confidence=1.0,
                    twenties_thirties_confidence=1.0,
                    backend="demo_age_bypass",
                )
                for _ in images
            ]
        probabilities = self.clip_ranker.classify(images, self.prompts)
        if probabilities is None:
            return [
                AgeAppearanceEstimate(
                    label="unknown",
                    twenties_confidence=0.0,
                    twenties_thirties_confidence=0.0,
                    backend="unavailable",
                )
                for _ in images
            ]
        return self.from_probabilities(probabilities)

    def from_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> list[AgeAppearanceEstimate]:
        return [
            AgeAppearanceEstimate(
                label=self.labels[int(np.argmax(row))],
                twenties_confidence=float(row[0]),
                twenties_thirties_confidence=float(row[0] + row[1]),
                backend="open_clip_zero_shot_shared_image_features",
            )
            for row in probabilities
        ]
