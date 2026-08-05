from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from core.utils import resolve_device


GENDER_LABELS = ("male", "female")
RACE_LABELS = (
    "east_asian",
    "southeast_asian",
    "white",
    "black",
    "indian",
    "middle_eastern",
    "latino_hispanic",
)


class DemographicClassifier(ABC):
    """Interface for operational gender and race probability estimators."""

    gender_labels = GENDER_LABELS
    race_labels = RACE_LABELS

    @abstractmethod
    def predict_proba(
        self, images: torch.Tensor | Sequence[Image.Image]
    ) -> dict[str, torch.Tensor]:
        """Return gender (N, 2) and race (N, 7) probability tensors."""
        raise NotImplementedError


class FairFaceDemographicClassifier(DemographicClassifier):
    """Lazy FairFace ResNet-34 classifier adapter."""

    _fairface_race_labels = (
        "white",
        "black",
        "latino_hispanic",
        "east_asian",
        "southeast_asian",
        "indian",
        "middle_eastern",
    )

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.device = resolve_device(device)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model = None

    @property
    def model(self):
        """Load FairFace weights only when inference is first requested."""
        if self._model is None:
            if not self.weights_path.exists():
                raise FileNotFoundError(
                    f"FairFace weights not found: {self.weights_path}. "
                    "Pass --fairface-weights or set demographic.weights."
                )
            from torchvision.models import resnet34

            model = resnet34(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 18)
            state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            cleaned = {
                key.removeprefix("module."): value for key, value in state.items()
            }
            model.load_state_dict(cleaned)
            self._model = model.eval().requires_grad_(False).to(self.device)
        return self._model

    def predict_proba(
        self, images: torch.Tensor | Sequence[Image.Image]
    ) -> dict[str, torch.Tensor]:
        """Predict labels using the standard FairFace 224-pixel preprocessing."""
        batch = self._prepare_images(images)
        gender_batches: list[torch.Tensor] = []
        race_batches: list[torch.Tensor] = []
        race_indices = [
            self._fairface_race_labels.index(label) for label in self.race_labels
        ]
        for start in range(0, len(batch), self.batch_size):
            values = batch[start : start + self.batch_size].to(self.device)
            with torch.no_grad():
                logits = self.model(values)
                race = torch.softmax(logits[:, :7], dim=1)[:, race_indices]
                gender = torch.softmax(logits[:, 7:9], dim=1)
            gender_batches.append(gender.cpu())
            race_batches.append(race.cpu())
        return {
            "gender": torch.cat(gender_batches, dim=0),
            "race": torch.cat(race_batches, dim=0),
        }

    @staticmethod
    def _prepare_images(
        images: torch.Tensor | Sequence[Image.Image],
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images.detach().float()
            if batch.ndim != 4 or batch.shape[1] != 3:
                raise ValueError("image tensor must have shape (N, 3, H, W)")
            if float(batch.min()) < 0:
                batch = (batch + 1.0) / 2.0
        else:
            arrays = []
            for image in images:
                if not isinstance(image, Image.Image):
                    raise TypeError("images must be a tensor or PIL Image sequence")
                arrays.append(
                    np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
                    / 255.0
                )
            if not arrays:
                raise ValueError("images must not be empty")
            batch = torch.from_numpy(np.stack(arrays))
        batch = torch.nn.functional.interpolate(
            batch, size=(224, 224), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        mean = torch.tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = torch.tensor((0.229, 0.224, 0.225))[None, :, None, None]
        return (batch.cpu() - mean) / std
