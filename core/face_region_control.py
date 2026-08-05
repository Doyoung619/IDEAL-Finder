from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from core.clip_ranker import CLIPRanker


@dataclass(frozen=True)
class FaceRegionEstimate:
    label: str
    east_asian_confidence: float
    backend: str


class FaceRegionController:
    prompts = (
        "a realistic portrait photograph of an East Asian person age 18 or older from "
        "Korea, China, or Japan",
        "a realistic portrait photograph of a White European person age 18 or older",
        "a realistic portrait photograph of a Black African person age 18 or older",
        "a realistic portrait photograph of a South Asian person age 18 or older from "
        "India or Pakistan",
        "a realistic portrait photograph of a Southeast Asian person age 18 or older",
        "a realistic portrait photograph of a Middle Eastern or Latin American "
        "person age 18 or older",
    )

    labels = (
        "east_asian",
        "white_european",
        "black_african",
        "south_asian",
        "southeast_asian",
        "middle_eastern_or_latin",
    )

    def __init__(self, clip_ranker: CLIPRanker) -> None:
        self.clip_ranker = clip_ranker

    def estimate(
        self,
        images: Sequence[Image.Image],
        generator_name: str,
    ) -> list[FaceRegionEstimate]:
        if generator_name.startswith("demo"):
            return [
                FaceRegionEstimate(
                    label="east_asian",
                    east_asian_confidence=1.0,
                    backend="demo_region_bypass",
                )
                for _ in images
            ]
        probabilities = self.clip_ranker.classify(images, self.prompts)
        if probabilities is None:
            return [
                FaceRegionEstimate(
                    label="unknown",
                    east_asian_confidence=0.0,
                    backend="unavailable",
                )
                for _ in images
            ]
        return self.from_probabilities(probabilities)

    def from_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> list[FaceRegionEstimate]:
        return [
            FaceRegionEstimate(
                label=self.labels[int(np.argmax(row))],
                east_asian_confidence=float(row[0]),
                backend="open_clip_zero_shot_shared_image_features",
            )
            for row in probabilities
        ]
