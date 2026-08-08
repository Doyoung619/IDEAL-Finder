from __future__ import annotations

import sys
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core.utils import resolve_device

try:
    import torch
except ImportError:  # pragma: no cover - exercised in Vercel slim installs.
    torch = None


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
            if torch is None:
                raise RuntimeError(
                    "Torch is required for StyleGAN2-ADA generation. "
                    "Use IDEAL_DEMO=1 for the lightweight web demo."
                )
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

    @property
    def z_dim(self) -> int:
        """Return the input dimension of the StyleGAN mapping network."""
        return int(self.network.z_dim)

    @property
    def w_dim(self) -> int:
        """Return the dimension of a single StyleGAN W latent."""
        return int(self.network.w_dim)

    def map_z_to_w(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Map a validated batch of Z vectors to single-vector W-space."""
        if isinstance(z, torch.Tensor):
            values = z.detach().to(dtype=torch.float32, device=self.device)
        else:
            array = np.asarray(z, dtype=np.float32)
            values = torch.from_numpy(array).to(self.device)
        if values.ndim != 2 or values.shape[1] != self.z_dim:
            raise ValueError(f"z must have shape (N, {self.z_dim})")
        if not torch.isfinite(values).all():
            raise ValueError("z contains NaN or Inf")
        condition = torch.zeros(
            (values.shape[0], self.network.c_dim), device=self.device
        )
        with torch.no_grad():
            w_plus = self.network.mapping(
                values,
                condition,
                truncation_psi=self.truncation_psi,
            )
        return w_plus[:, 0, :].detach().cpu().numpy().astype(np.float32)

    def synthesize_w(self, w: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Synthesize a W batch as CPU float images in the [0, 1] range."""
        if isinstance(w, torch.Tensor):
            values = w.detach().to(dtype=torch.float32, device=self.device)
        else:
            values = torch.from_numpy(np.asarray(w, dtype=np.float32)).to(
                self.device
            )
        if values.ndim != 2 or values.shape[1] != self.w_dim:
            raise ValueError(f"w must have shape (N, {self.w_dim})")
        if not torch.isfinite(values).all():
            raise ValueError("w contains NaN or Inf")
        image_batches: list[torch.Tensor] = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            w_plus = batch.unsqueeze(1).repeat(1, self.network.num_ws, 1)
            with torch.no_grad():
                images = self.network.synthesis(
                    w_plus,
                    noise_mode=self.noise_mode,
                    force_fp32=True,
                )
            image_batches.append(((images + 1.0) / 2.0).clamp(0, 1).cpu())
        if not image_batches:
            return torch.empty((0, 3, self.output_resolution, self.output_resolution))
        return torch.cat(image_batches, dim=0)

    def generate_from_theta(self, theta: np.ndarray, prior) -> torch.Tensor:
        """Map conditional PCA coordinates to W and synthesize their images."""
        w = np.atleast_2d(prior.theta_to_w(theta))
        if w.shape[1] != self.w_dim:
            raise ValueError(
                f"prior W dimension {w.shape[1]} does not match generator {self.w_dim}"
            )
        return self.synthesize_w(w)

    def sample_prior(self, count: int, seed: int) -> np.ndarray:
        random = np.random.default_rng(seed)
        z = random.standard_normal((count, self.z_dim)).astype(np.float32)
        return self.map_z_to_w(z)

    def decode(self, latents: np.ndarray) -> list[Image.Image]:
        latent_batch = np.atleast_2d(latents).astype(np.float32)
        tensors = self.synthesize_w(latent_batch)
        results: list[Image.Image] = []
        arrays = tensors.mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
        for array in arrays.numpy():
            image = Image.fromarray(array, "RGB")
            if image.size != (self.output_resolution, self.output_resolution):
                image = image.resize(
                    (self.output_resolution, self.output_resolution),
                    Image.Resampling.LANCZOS,
                )
            results.append(image)
        return results


class DemoFaceGenerator(FaceGenerator):
    generator_name = "demo_numpy_portrait_decoder"

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
        latent_batch = np.atleast_2d(latents).astype(np.float32)
        return [self._render_portrait(latent) for latent in latent_batch]

    def _render_portrait(self, latent: np.ndarray) -> Image.Image:
        size = int(self.output_resolution)
        image = Image.new(
            "RGB",
            (size, size),
            self._color(latent, (1, 2, 3), (158, 171, 184)),
        )
        draw = ImageDraw.Draw(image, "RGBA")

        def v(index: int, scale: float = 1.0) -> float:
            return float(np.tanh(latent[index % self.latent_dim]) * scale)

        def point(x: float, y: float) -> tuple[int, int]:
            return (
                int((x + 1.0) * 0.5 * size),
                int((y + 1.0) * 0.5 * size),
            )

        def ellipse(
            center_x: float,
            center_y: float,
            radius_x: float,
            radius_y: float,
            fill,
        ) -> None:
            left, top = point(center_x - radius_x, center_y - radius_y)
            right, bottom = point(center_x + radius_x, center_y + radius_y)
            draw.ellipse((left, top, right, bottom), fill=fill)

        gender_axis = v(0)
        shoulder = self._color(latent, (20, 21, 22), (40, 50, 62))
        skin = self._color(latent, (4, 5, 6), (170, 130, 102))
        hair = self._color(latent, (10, 11, 12), (28, 24, 22))
        lip = self._color(latent, (26, 27, 28), (142, 57, 62))

        ellipse(0.0, 0.98, 0.86, 0.42, (*shoulder, 255))
        ellipse(0.0, 0.58, 0.19, 0.32, (*skin, 255))

        face_width = 0.48 + 0.06 * v(7) + 0.03 * gender_axis
        face_height = 0.67 + 0.04 * v(8)
        face_x = v(9, 0.025)
        face_y = -0.03
        hair_height = 0.80 + 0.08 * v(13) - 0.12 * gender_axis

        ellipse(face_x, face_y - 0.10, face_width + 0.10, hair_height, (*hair, 255))
        ellipse(face_x, face_y, face_width, face_height, (*skin, 255))

        fringe_x = face_x + v(14, 0.08)
        ellipse(fringe_x, -0.56, 0.42, 0.18, (*hair, 235))

        eye_spacing = 0.19 + 0.025 * v(16)
        eye_y = -0.13 + 0.025 * v(17)
        eye_width = 0.055 + 0.012 * v(18)
        ellipse(-eye_spacing, eye_y, eye_width, 0.032, (25, 25, 28, 255))
        ellipse(eye_spacing, eye_y, eye_width, 0.032, (25, 25, 28, 255))
        ellipse(-eye_spacing, eye_y - 0.09, 0.13, 0.018, (*hair, 220))
        ellipse(eye_spacing, eye_y - 0.09, 0.13, 0.018, (*hair, 220))

        nose_x = v(19, 0.018)
        nose_top = point(nose_x, -0.03)
        nose_bottom = point(nose_x + v(22, 0.018), 0.18)
        draw.line(
            (nose_top, nose_bottom),
            fill=(*self._shade(skin, 0.78), 120),
            width=max(1, size // 55),
        )

        mouth_y = 0.32 + v(23, 0.035)
        mouth_width = 0.16 + 0.035 * v(24)
        ellipse(v(25, 0.018), mouth_y, mouth_width, 0.030, (*lip, 210))
        ellipse(-0.30, 0.18, 0.13, 0.08, (230, 112, 112, 30))
        ellipse(0.30, 0.18, 0.13, 0.08, (230, 112, 112, 30))

        return image

    def _color(
        self,
        latent: np.ndarray,
        indices: tuple[int, int, int],
        base: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        values = []
        for index, channel in zip(indices, base):
            offset = int(
                (
                    1.0
                    / (1.0 + np.exp(-latent[index % self.latent_dim]))
                    - 0.5
                )
                * 46
            )
            values.append(int(np.clip(channel + offset, 0, 255)))
        return tuple(values)

    @staticmethod
    def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
        return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


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


StyleGAN2GeneratorAdapter = StyleGAN2ADAGenerator
