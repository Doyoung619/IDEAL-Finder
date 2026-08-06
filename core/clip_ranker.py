from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from core.utils import resolve_device


class CLIPRanker:
    def __init__(
        self,
        enabled: bool,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.backend = "disabled"
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def _text_features(self, prompts: Sequence[str]) -> torch.Tensor:
        key = tuple(prompts)
        cached = self._text_features_cache.get(key)
        if cached is not None:
            return cached
        text_tensor = self._tokenizer(list(key)).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(text_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        self._text_features_cache[key] = features
        return features

    def _ensure_loaded(self) -> bool:
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        try:
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            self._model = model.eval()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self.backend = "open_clip"
            return True
        except (ImportError, RuntimeError, OSError):
            self.enabled = False
            self.backend = "latent_hash_fallback"
            return False

    def score(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        latent_features: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._ensure_loaded():
            text_features = self._text_features([prompt])
            with torch.no_grad():
                score_batches = []
                for start in range(0, len(images), self.batch_size):
                    image_tensor = torch.stack(
                        [
                            self._preprocess(image.convert("RGB"))
                            for image in images[start : start + self.batch_size]
                        ]
                    ).to(self.device)
                    image_features = self._model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    score_batches.append(
                        (image_features @ text_features.T)[:, 0].float().cpu()
                    )
            return torch.cat(score_batches).numpy()

        self.backend = "latent_hash_fallback"
        if latent_features is None:
            return np.asarray(
                [
                    np.asarray(image.convert("RGB"), dtype=np.float32).std() / 255.0
                    for image in images
                ],
                dtype=np.float32,
            )
        features = np.atleast_2d(latent_features).astype(np.float32)
        seed = int.from_bytes(
            hashlib.sha256(prompt.encode("utf-8")).digest()[:8],
            "big",
        )
        random = np.random.default_rng(seed)
        direction = random.standard_normal(features.shape[1]).astype(np.float32)
        direction /= np.linalg.norm(direction) + 1e-8
        return (features @ direction).astype(np.float32)

    def classify(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
    ) -> np.ndarray | None:
        if not self._ensure_loaded():
            return None
        text_features = self._text_features(prompts)
        with torch.no_grad():
            probability_batches = []
            for start in range(0, len(images), self.batch_size):
                image_tensor = torch.stack(
                    [
                        self._preprocess(image.convert("RGB"))
                        for image in images[start : start + self.batch_size]
                    ]
                ).to(self.device)
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                probability_batches.append(
                    (100.0 * image_features @ text_features.T)
                    .softmax(dim=-1)
                    .float()
                    .cpu()
                )
        return torch.cat(probability_batches).numpy()

    def classify_groups(
        self,
        images: Sequence[Image.Image],
        prompt_groups: Sequence[Sequence[str]],
    ) -> list[np.ndarray] | None:
        if not self._ensure_loaded():
            return None
        flat_prompts = [
            prompt for group in prompt_groups for prompt in group
        ]
        text_features = self._text_features(flat_prompts)
        with torch.no_grad():
            image_feature_batches = []
            for start in range(0, len(images), self.batch_size):
                image_tensor = torch.stack(
                    [
                        self._preprocess(image.convert("RGB"))
                        for image in images[start : start + self.batch_size]
                    ]
                ).to(self.device)
                image_features = self._model.encode_image(image_tensor)
                image_feature_batches.append(
                    (
                        image_features
                        / image_features.norm(dim=-1, keepdim=True)
                    ).float()
                )
            all_image_features = torch.cat(image_feature_batches)
            logits = 100.0 * all_image_features @ text_features.T
            results: list[np.ndarray] = []
            offset = 0
            for group in prompt_groups:
                width = len(group)
                results.append(
                    logits[:, offset : offset + width]
                    .softmax(dim=-1)
                    .cpu()
                    .numpy()
                )
                offset += width
        return results
