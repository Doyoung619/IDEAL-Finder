from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import text

from app.db import get_session


def validate_gpu_startup(config, runtime) -> dict[str, object]:
    """Fail early when a real GPU worker cannot serve the configured study."""
    with get_session() as db:
        db.execute(text("SELECT 1"))

    if config.generator.mode == "demo":
        return {
            "database": "ok",
            "cuda": False,
            "generator_loaded": True,
            "openclip_loaded": False,
            "artifacts": "mock",
        }

    admin_password = os.getenv("IDEAL_ADMIN_PASSWORD", "")
    if not admin_password:
        raise RuntimeError(
            "IDEAL_ADMIN_PASSWORD is required for the real GPU worker"
        )
    if len(admin_password) < 16:
        raise RuntimeError(
            "IDEAL_ADMIN_PASSWORD must contain at least 16 characters"
        )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for APP_ROLE=gpu research mode") from exc

    configured_devices = {
        str(config.generator.device),
        str(config.clip.device),
    }
    if any(device.startswith("cuda") for device in configured_devices):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable, but the GPU configuration requests a CUDA device"
            )

    _require_path(config.generator.stylegan_repo, "StyleGAN source directory")
    _require_path(config.generator.network_path, "StyleGAN checkpoint")

    # Loading latent_dim materializes the checkpoint and validates its W shape.
    _ = runtime.generator.latent_dim
    for gender in ("female", "male"):
        runtime.persona_pool(gender)
    if bool(config.clip.enabled) and not runtime.clip_ranker.ensure_ready():
        raise RuntimeError("The configured OpenCLIP model could not be loaded")

    return {
        "database": "ok",
        "cuda": bool(torch.cuda.is_available()),
        "generator_loaded": True,
        "openclip_loaded": bool(config.clip.enabled),
        "artifacts": "ok",
    }


def _require_path(value: str, label: str) -> None:
    if not Path(value).exists():
        raise RuntimeError(f"{label} does not exist: {value}")
