from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from core.generator import FaceGenerator


class LatentProjector:
    def __init__(self, model_path: str | Path, dimensions: int = 32) -> None:
        self.model_path = Path(model_path)
        self.dimensions = dimensions
        self.model: PCA | None = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def fit(self, latents: np.ndarray) -> "LatentProjector":
        samples = np.atleast_2d(latents).astype(np.float32)
        components = min(self.dimensions, samples.shape[0] - 1, samples.shape[1])
        if components < 1:
            raise ValueError("At least two latent samples are required to fit PCA")
        self.model = PCA(
            n_components=components,
            whiten=True,
            svd_solver="randomized",
            random_state=0,
        )
        self.model.fit(samples)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        return self

    def ensure_fitted(
        self,
        generator: FaceGenerator,
        sample_count: int,
        seed: int,
    ) -> "LatentProjector":
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            return self
        minimum_samples = max(sample_count, self.dimensions + 1)
        return self.fit(generator.sample_prior(minimum_samples, seed))

    def transform(self, latents: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA projector is not fitted")
        return self.model.transform(np.atleast_2d(latents)).astype(np.float32)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA projector is not fitted")
        return self.model.inverse_transform(np.atleast_2d(features)).astype(np.float32)

