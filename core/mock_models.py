from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from core.demographic_classifier import DemographicClassifier


class MockStyleGANGenerator:
    """Small deterministic StyleGAN-like generator for CPU-only tests and demos."""

    generator_name = "mock_stylegan"

    def __init__(self, w_dimension: int = 8, image_size: int = 16) -> None:
        self.z_dim = int(w_dimension)
        self.w_dim = int(w_dimension)
        self.latent_dim = int(w_dimension)
        self.image_size = int(image_size)
        self.truncation_psi = 1.0

    def map_z_to_w(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Apply a deterministic, full-rank nonlinear mock mapping."""
        values = np.asarray(z, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.z_dim:
            raise ValueError(f"z must have shape (N, {self.z_dim})")
        return (0.85 * values + 0.15 * np.tanh(values)).astype(np.float32)

    def synthesize_w(self, w: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode the first W coordinates into a deterministic RGB tensor."""
        values = np.asarray(w, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.w_dim:
            raise ValueError(f"w must have shape (N, {self.w_dim})")
        channels = torch.sigmoid(torch.from_numpy(values[:, :3]))
        return channels[:, :, None, None].repeat(
            1, 1, self.image_size, self.image_size
        )

    def sample_prior(self, count: int, seed: int) -> np.ndarray:
        """Sample mock W values through the mock mapping network."""
        random = np.random.default_rng(seed)
        return self.map_z_to_w(
            random.standard_normal((count, self.z_dim)).astype(np.float32)
        )

    def decode(self, latents: np.ndarray) -> list[Image.Image]:
        """Decode mock W vectors to PIL images."""
        tensors = self.synthesize_w(np.atleast_2d(latents))
        return [
            Image.fromarray(
                tensor.mul(255).byte().permute(1, 2, 0).numpy(), "RGB"
            )
            for tensor in tensors
        ]

    def generate_from_theta(self, theta: np.ndarray, prior) -> torch.Tensor:
        """Map theta through a conditional prior and synthesize mock images."""
        return self.synthesize_w(prior.theta_to_w(theta))


class MockDemographicClassifier(DemographicClassifier):
    """Deterministic classifier whose feasible region is controlled by RGB means."""

    def __init__(self, sharpness: float = 8.0) -> None:
        self.sharpness = float(sharpness)

    def predict_proba(
        self, images: torch.Tensor | Sequence[Image.Image]
    ) -> dict[str, torch.Tensor]:
        """Treat bright red as female and bright green as East Asian."""
        if isinstance(images, torch.Tensor):
            batch = images.detach().float().cpu()
        else:
            arrays = [
                np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
                / 255.0
                for image in images
            ]
            batch = torch.from_numpy(np.stack(arrays))
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError("image tensor must have shape (N, 3, H, W)")
        means = batch.mean(dim=(2, 3))
        female = torch.sigmoid(self.sharpness * (means[:, 0] - 0.5))
        east_asian = torch.sigmoid(self.sharpness * (means[:, 1] - 0.5))
        gender = torch.stack((1.0 - female, female), dim=1)
        remaining = (1.0 - east_asian) / 6.0
        race = torch.stack(
            (
                east_asian,
                remaining,
                remaining,
                remaining,
                remaining,
                remaining,
                remaining,
            ),
            dim=1,
        )
        return {"gender": gender, "race": race}
