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
AGE_LABELS = (
    "0_2",
    "3_9",
    "10_19",
    "20_29",
    "30_39",
    "40_49",
    "50_59",
    "60_69",
    "70_plus",
)


class DemographicClassifier(ABC):
    """Interface for operational gender and race probability estimators."""

    gender_labels = GENDER_LABELS
    race_labels = RACE_LABELS
    age_labels = AGE_LABELS

    @abstractmethod
    def predict_proba(
        self, images: torch.Tensor | Sequence[Image.Image]
    ) -> dict[str, torch.Tensor]:
        """Return gender (N, 2), race (N, 7), and age (N, 9) probabilities."""
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
        age_batches: list[torch.Tensor] = []
        race_indices = [
            self._fairface_race_labels.index(label) for label in self.race_labels
        ]
        for start in range(0, len(batch), self.batch_size):
            values = batch[start : start + self.batch_size].to(self.device)
            with torch.no_grad():
                logits = self.model(values)
                race = torch.softmax(logits[:, :7], dim=1)[:, race_indices]
                gender = torch.softmax(logits[:, 7:9], dim=1)
                age = torch.softmax(logits[:, 9:18], dim=1)
            gender_batches.append(gender.cpu())
            race_batches.append(race.cpu())
            age_batches.append(age.cpu())
        return {
            "gender": torch.cat(gender_batches, dim=0),
            "race": torch.cat(race_batches, dim=0),
            "age": torch.cat(age_batches, dim=0),
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


class FairFaceViTDemographicClassifier(DemographicClassifier):
    """Three locally cached FairFace-trained ViT classifiers."""

    _race_reorder = (0, 6, 3, 2, 1, 4, 5)
    _gender_reorder = (1, 0)

    def __init__(
        self,
        race_model_path: str | Path,
        gender_model_path: str | Path,
        age_model_path: str | Path,
        devices: Sequence[str] | None = None,
        batch_size: int = 32,
        use_fp16: bool = True,
    ) -> None:
        from transformers import ViTForImageClassification

        paths = [Path(race_model_path), Path(gender_model_path), Path(age_model_path)]
        missing = [str(path) for path in paths if not (path / "model.safetensors").exists()]
        if missing:
            raise FileNotFoundError(f"Missing FairFace ViT checkpoints: {missing}")
        values = list(devices or (["cpu"] * 3))
        if len(values) != 3:
            raise ValueError("devices must contain race, gender, and age devices")
        self.devices = [torch.device(resolve_device(value)) for value in values]
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16 and all(device.type == "cuda" for device in self.devices))
        self.models = []
        for path, device in zip(paths, self.devices):
            model = ViTForImageClassification.from_pretrained(
                str(path), local_files_only=True
            ).eval().requires_grad_(False).to(device)
            if self.use_fp16:
                model = model.half()
            self.models.append(model)

    def predict_proba(
        self, images: torch.Tensor | Sequence[Image.Image]
    ) -> dict[str, torch.Tensor]:
        batch = self._prepare_images(images)
        race_batches: list[torch.Tensor] = []
        gender_batches: list[torch.Tensor] = []
        age_batches: list[torch.Tensor] = []
        for start in range(0, len(batch), self.batch_size):
            values = batch[start : start + self.batch_size]
            inputs = []
            for device in self.devices:
                item = values.to(device, non_blocking=True)
                inputs.append(item.half() if self.use_fp16 else item)
            with torch.inference_mode():
                race = self.models[0](pixel_values=inputs[0]).logits.float().softmax(dim=-1)
                gender = self.models[1](pixel_values=inputs[1]).logits.float().softmax(dim=-1)
                age = self.models[2](pixel_values=inputs[2]).logits.float().softmax(dim=-1)
            race_batches.append(race[:, self._race_reorder].cpu())
            gender_batches.append(gender[:, self._gender_reorder].cpu())
            age_batches.append(age.cpu())
        return {
            "gender": torch.cat(gender_batches),
            "race": torch.cat(race_batches),
            "age": torch.cat(age_batches),
        }

    @staticmethod
    def _prepare_images(
        images: torch.Tensor | Sequence[Image.Image],
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images.detach().float()
            if batch.ndim != 4 or batch.shape[1] != 3:
                raise ValueError("image tensor must have shape (N, 3, H, W)")
            if float(batch.min()) >= 0:
                batch = batch * 2.0 - 1.0
        else:
            arrays = [
                np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
                / 127.5
                - 1.0
                for image in images
            ]
            if not arrays:
                raise ValueError("images must not be empty")
            batch = torch.from_numpy(np.stack(arrays))
        return torch.nn.functional.interpolate(
            batch, size=(224, 224), mode="bilinear", align_corners=False
        ).clamp(-1.0, 1.0)
