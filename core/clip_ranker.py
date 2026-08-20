from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from PIL import Image

from core.utils import resolve_device

try:
    import torch
except ImportError:  # pragma: no cover - exercised in Vercel slim installs.
    torch = None


class CLIPRanker:
    def __init__(
        self,
        enabled: bool,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        batch_size: int = 16,
        require_real: bool = False,
        allow_mock: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.require_real = bool(require_real)
        self.allow_mock = bool(allow_mock)
        self.backend = "disabled"
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._load_error: Exception | None = None

    def ensure_ready(self) -> bool:
        """Materialize the configured model for production startup validation."""
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if not self.enabled or torch is None:
            if torch is None:
                self.enabled = False
                self.backend = "latent_hash_fallback"
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
        except (ImportError, RuntimeError, OSError) as exc:
            self._load_error = exc
            self.enabled = False
            self.backend = "latent_hash_fallback"
            if self.require_real:
                raise RuntimeError(
                    "Persona retrieval requires a real OpenCLIP model. "
                    "Install/download the configured checkpoint before research "
                    "data collection."
                ) from exc
            return False

    @staticmethod
    def _normalized(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        if not np.isfinite(array).all():
            raise ValueError("embeddings contain NaN or Inf")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("embeddings must have non-zero norms")
        return (array / norms).astype(np.float32)

    @staticmethod
    def _mock_embedding(payload: bytes, dimensions: int = 64) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return np.random.default_rng(seed).standard_normal(dimensions).astype(np.float32)

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return L2-normalized float32 image embeddings."""
        if not images:
            raise ValueError("images must not be empty")
        if self._ensure_loaded():
            batches = []
            with torch.no_grad():
                for start in range(0, len(images), self.batch_size):
                    tensor = torch.stack(
                        [self._preprocess(image.convert("RGB")) for image in images[start : start + self.batch_size]]
                    ).to(self.device)
                    batches.append(self._model.encode_image(tensor).float().cpu().numpy())
            return self._normalized(np.concatenate(batches, axis=0))
        if not self.allow_mock:
            raise RuntimeError(
                "Persona retrieval requires a real OpenCLIP model. "
                "Use demo mode or explicitly enable mock embeddings for tests."
            )
        self.backend = "mock_deterministic"
        values = [
            self._mock_embedding(image.convert("RGB").tobytes())
            for image in images
        ]
        return self._normalized(np.stack(values))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized float32 text embeddings."""
        if not texts or any(not str(text).strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if self._ensure_loaded():
            with torch.no_grad():
                tensor = self._tokenizer(list(texts)).to(self.device)
                values = self._model.encode_text(tensor).float().cpu().numpy()
            return self._normalized(values)
        if not self.allow_mock:
            raise RuntimeError(
                "Persona retrieval requires a real OpenCLIP model. "
                "Use demo mode or explicitly enable mock embeddings for tests."
            )
        self.backend = "mock_deterministic"
        return self._normalized(
            np.stack([self._mock_embedding(str(value).encode("utf-8")) for value in texts])
        )

    def score(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        latent_features: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._ensure_loaded():
            text_tensor = self._tokenizer([prompt]).to(self.device)
            with torch.no_grad():
                text_features = self._model.encode_text(text_tensor)
                text_features = text_features / text_features.norm(
                    dim=-1, keepdim=True
                )
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
        text_tensor = self._tokenizer(list(prompts)).to(self.device)
        with torch.no_grad():
            text_features = self._model.encode_text(text_tensor)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
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
        text_tensor = self._tokenizer(flat_prompts).to(self.device)
        with torch.no_grad():
            text_features = self._model.encode_text(text_tensor)
            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )
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
