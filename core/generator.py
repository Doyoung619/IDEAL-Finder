from __future__ import annotations

import sys
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from core.utils import resolve_device


class FaceGenerator(ABC):
    generator_name = "base"

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample_prior(self, count: int, seed: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latents: np.ndarray) -> list[Image.Image]:
        raise NotImplementedError


class StyleGAN2ADAGenerator(FaceGenerator):
    generator_name = "stylegan2_ada_ffhq"

    def __init__(
        self,
        repo_path: str,
        network_path: str,
        device: str = "auto",
        output_resolution: int = 256,
        batch_size: int = 8,
        truncation_psi: float = 0.7,
        noise_mode: str = "const",
    ) -> None:
        self.repo_path = Path(repo_path)
        self.network_path = Path(network_path)
        self.generator_name = f"stylegan2_ada_{self.network_path.stem}"
        self.device = resolve_device(device)
        self.output_resolution = output_resolution
        self.batch_size = batch_size
        self.truncation_psi = truncation_psi
        self.noise_mode = noise_mode
        self._network = None

    @property
    def network(self):
        if self._network is None:
            if not self.repo_path.exists():
                raise FileNotFoundError(
                    f"StyleGAN2-ADA source not found: {self.repo_path}. "
                    "Run scripts/download_models.py first."
                )
            if not self.network_path.exists():
                raise FileNotFoundError(
                    f"FFHQ network not found: {self.network_path}. "
                    "Run scripts/download_models.py first."
                )
            sys.path.insert(0, str(self.repo_path))
            try:
                import legacy

                self._configure_windows_reference_ops()
                with self.network_path.open("rb") as handle:
                    self._network = legacy.load_network_pkl(handle)["G_ema"]
                self._network = self._network.eval().requires_grad_(False).to(self.device)
            finally:
                if sys.path[0] == str(self.repo_path):
                    sys.path.pop(0)
        return self._network

    @staticmethod
    def _configure_windows_reference_ops() -> None:
        if os.name != "nt" or shutil.which("cl") is not None:
            return
        from torch_utils.ops import bias_act, upfirdn2d

        functions = (
            bias_act.bias_act,
            upfirdn2d.upfirdn2d,
            upfirdn2d.filter2d,
            upfirdn2d.upsample2d,
            upfirdn2d.downsample2d,
        )
        for function in functions:
            defaults = list(function.__defaults__ or ())
            if defaults and defaults[-1] == "cuda":
                defaults[-1] = "ref"
                function.__defaults__ = tuple(defaults)

    @property
    def latent_dim(self) -> int:
        return int(self.network.w_dim)

    def sample_prior(self, count: int, seed: int) -> np.ndarray:
        random = np.random.default_rng(seed)
        z = torch.from_numpy(
            random.standard_normal((count, self.network.z_dim)).astype(np.float32)
        ).to(self.device)
        c = torch.zeros((count, self.network.c_dim), device=self.device)
        with torch.no_grad():
            w_plus = self.network.mapping(
                z,
                c,
                truncation_psi=self.truncation_psi,
            )
        return w_plus[:, 0, :].detach().cpu().numpy().astype(np.float32)

    def decode(self, latents: np.ndarray) -> list[Image.Image]:
        latent_batch = np.atleast_2d(latents).astype(np.float32)
        results: list[Image.Image] = []
        for start in range(0, len(latent_batch), self.batch_size):
            batch = latent_batch[start : start + self.batch_size]
            w = torch.from_numpy(batch).to(self.device)
            w_plus = w.unsqueeze(1).repeat(1, self.network.num_ws, 1)
            with torch.no_grad():
                images = self.network.synthesis(
                    w_plus,
                    noise_mode=self.noise_mode,
                    force_fp32=True,
                )
                images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                images = images.permute(0, 2, 3, 1).cpu().numpy()
            for array in images:
                image = Image.fromarray(array, "RGB")
                if image.size != (self.output_resolution, self.output_resolution):
                    image = image.resize(
                        (self.output_resolution, self.output_resolution),
                        Image.Resampling.LANCZOS,
                    )
                results.append(image)
        return results


class DemoFaceGenerator(FaceGenerator):
    generator_name = "demo_torch_portrait_decoder"

    def __init__(
        self,
        latent_dim: int = 32,
        output_resolution: int = 256,
        device: str = "auto",
    ) -> None:
        self._latent_dim = latent_dim
        self.output_resolution = output_resolution
        self.device = resolve_device(device)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def sample_prior(self, count: int, seed: int) -> np.ndarray:
        random = np.random.default_rng(seed)
        return random.standard_normal((count, self.latent_dim)).astype(np.float32)

    def decode(self, latents: np.ndarray) -> list[Image.Image]:
        latent_batch = torch.from_numpy(
            np.atleast_2d(latents).astype(np.float32)
        ).to(self.device)
        with torch.no_grad():
            images = self._render_portraits(latent_batch)
            images = (
                images.mul(255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            )
        return [
            Image.fromarray(array.cpu().numpy(), "RGB")
            for array in images
        ]

    def _render_portraits(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        size = self.output_resolution
        coordinates = torch.linspace(-1.0, 1.0, size, device=self.device)
        y_grid, x_grid = torch.meshgrid(coordinates, coordinates, indexing="ij")
        x_grid = x_grid.expand(batch, -1, -1)
        y_grid = y_grid.expand(batch, -1, -1)

        def value(index: int, scale: float = 1.0) -> torch.Tensor:
            return torch.tanh(latent[:, index % self.latent_dim]) * scale

        def color(red: int, green: int, blue: int, base: tuple[float, float, float]):
            channels = [
                torch.sigmoid(latent[:, index % self.latent_dim]) * 0.18 + offset
                for index, offset in zip((red, green, blue), base)
            ]
            return torch.stack(channels, dim=1).clamp(0.0, 1.0)

        def ellipse(
            center_x: torch.Tensor,
            center_y: torch.Tensor,
            radius_x: torch.Tensor,
            radius_y: torch.Tensor,
            softness: float = 45.0,
        ) -> torch.Tensor:
            distance = (
                ((x_grid - center_x[:, None, None]) / radius_x[:, None, None]) ** 2
                + ((y_grid - center_y[:, None, None]) / radius_y[:, None, None]) ** 2
            )
            return torch.sigmoid((1.0 - distance) * softness)

        def gaussian(
            center_x: torch.Tensor,
            center_y: torch.Tensor,
            width: float,
            height: float,
        ) -> torch.Tensor:
            return torch.exp(
                -(
                    ((x_grid - center_x[:, None, None]) / width) ** 2
                    + ((y_grid - center_y[:, None, None]) / height) ** 2
                )
                * 3.0
            )

        background = color(1, 2, 3, (0.62, 0.67, 0.72))
        image = background[:, :, None, None].expand(-1, -1, size, size).clone()

        shoulder_color = color(20, 21, 22, (0.12, 0.18, 0.24))
        shoulder_mask = ellipse(
            torch.zeros(batch, device=self.device),
            torch.full((batch,), 0.92, device=self.device),
            torch.full((batch,), 0.86, device=self.device),
            torch.full((batch,), 0.46, device=self.device),
        )
        image = self._blend(image, shoulder_color, shoulder_mask)

        skin = color(4, 5, 6, (0.58, 0.43, 0.34))
        neck_mask = ellipse(
            torch.zeros(batch, device=self.device),
            torch.full((batch,), 0.58, device=self.device),
            torch.full((batch,), 0.20, device=self.device),
            torch.full((batch,), 0.34, device=self.device),
        )
        image = self._blend(image, skin, neck_mask)

        gender_axis = value(0)
        face_width = 0.49 + 0.06 * value(7) + 0.035 * gender_axis
        face_height = 0.68 + 0.04 * value(8)
        face_y = torch.full((batch,), -0.03, device=self.device)
        face_mask = ellipse(
            value(9, 0.025),
            face_y,
            face_width,
            face_height,
        )
        image = self._blend(image, skin, face_mask)

        hair_color = color(10, 11, 12, (0.05, 0.035, 0.025))
        hair_length = 0.10 - 0.18 * gender_axis + 0.08 * value(13)
        outer_hair = ellipse(
            value(9, 0.02),
            face_y - 0.09,
            face_width + 0.09,
            face_height + 0.12 + hair_length.clamp(-0.08, 0.25),
        )
        lower_cut = torch.sigmoid((-y_grid + 0.68 + hair_length[:, None, None]) * 40)
        hair_mask = outer_hair * lower_cut * (1.0 - face_mask * 0.86)
        fringe = gaussian(
            value(14, 0.08),
            torch.full((batch,), -0.54, device=self.device),
            0.42,
            0.24,
        )
        hair_mask = torch.maximum(hair_mask, fringe * (0.65 + 0.25 * value(15))[:, None, None])
        image = self._blend(image, hair_color, hair_mask.clamp(0, 1))

        eye_spacing = 0.19 + 0.025 * value(16)
        eye_y = -0.13 + 0.025 * value(17)
        eye_size = 0.045 + 0.012 * value(18) - 0.004 * gender_axis
        left_eye = gaussian(-eye_spacing, eye_y, float(eye_size.mean()), 0.035)
        right_eye = gaussian(eye_spacing, eye_y, float(eye_size.mean()), 0.035)
        eye_mask = (left_eye + right_eye).clamp(0, 1) * face_mask
        image = self._blend(
            image,
            torch.full((batch, 3), 0.08, device=self.device),
            eye_mask,
        )

        brow_y = eye_y - 0.095
        brow_mask = (
            gaussian(-eye_spacing, brow_y, 0.13, 0.025)
            + gaussian(eye_spacing, brow_y, 0.13, 0.025)
        ).clamp(0, 1)
        image = self._blend(image, hair_color * 0.72, brow_mask * face_mask)

        nose = gaussian(
            value(19, 0.018),
            torch.full((batch,), 0.08, device=self.device),
            0.045,
            0.17,
        )
        nose_color = (skin * 0.82).clamp(0, 1)
        image = self._blend(image, nose_color, nose * 0.18 * face_mask)

        smile = value(23, 0.035)
        mouth_y = 0.32 + smile
        mouth_width = 0.16 + 0.035 * value(24)
        mouth = gaussian(
            value(25, 0.018),
            mouth_y,
            float(mouth_width.mean()),
            0.032,
        )
        lip_color = color(26, 27, 28, (0.34, 0.12, 0.13))
        image = self._blend(image, lip_color, mouth * face_mask * 0.8)

        cheek = (
            gaussian(
                torch.full((batch,), -0.30, device=self.device),
                torch.full((batch,), 0.18, device=self.device),
                0.16,
                0.11,
            )
            + gaussian(
                torch.full((batch,), 0.30, device=self.device),
                torch.full((batch,), 0.18, device=self.device),
                0.16,
                0.11,
            )
        ).clamp(0, 1)
        image = self._blend(image, torch.tensor([[0.82, 0.40, 0.38]], device=self.device).repeat(batch, 1), cheek * 0.08)

        return functional.interpolate(
            image,
            size=(self.output_resolution, self.output_resolution),
            mode="bilinear",
            align_corners=False,
        ).clamp(0, 1)

    @staticmethod
    def _blend(
        image: torch.Tensor,
        color: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        alpha = mask[:, None, :, :].clamp(0, 1)
        return image * (1 - alpha) + color[:, :, None, None] * alpha


def create_generator(config) -> FaceGenerator:
    if config.generator.mode == "demo":
        return DemoFaceGenerator(
            latent_dim=config.generator.demo_latent_dim,
            output_resolution=config.generator.output_resolution,
            device=config.generator.device,
        )
    if config.generator.mode == "stylegan2_ada":
        return StyleGAN2ADAGenerator(
            repo_path=config.generator.stylegan_repo,
            network_path=config.generator.network_path,
            device=config.generator.device,
            output_resolution=config.generator.output_resolution,
            batch_size=config.generator.batch_size,
            truncation_psi=config.generator.truncation_psi,
            noise_mode=config.generator.noise_mode,
        )
    raise ValueError(f"Unsupported generator mode: {config.generator.mode}")
