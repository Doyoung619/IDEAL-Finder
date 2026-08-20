from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from app.models import ExperimentBlock, LatentImage


def persist_artifacts_in_db(config) -> bool:
    storage = getattr(config, "storage", None)
    return bool(getattr(storage, "persist_artifacts_in_db", False))


def array_to_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array, dtype=np.float32))
    return buffer.getvalue()


def array_from_npy_bytes(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


def image_to_png_bytes(
    image: Image.Image,
    *,
    optimize: bool = True,
    compress_level: int | None = None,
) -> bytes:
    buffer = io.BytesIO()
    options = {"optimize": optimize}
    if compress_level is not None:
        options["compress_level"] = int(compress_level)
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


def write_file_if_missing(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def ensure_block_files(block: ExperimentBlock) -> None:
    if block.mu_data and not Path(block.mu_path).exists():
        write_file_if_missing(block.mu_path, block.mu_data)
    if (
        block.strategy_state_path
        and block.strategy_state_data
        and not Path(block.strategy_state_path).exists()
    ):
        write_file_if_missing(block.strategy_state_path, block.strategy_state_data)


def sync_block_files(block: ExperimentBlock) -> None:
    if Path(block.mu_path).exists():
        block.mu_data = Path(block.mu_path).read_bytes()
    if block.strategy_state_path and Path(block.strategy_state_path).exists():
        block.strategy_state_data = Path(block.strategy_state_path).read_bytes()


def load_latent(artifact: LatentImage) -> np.ndarray:
    path = Path(artifact.latent_path)
    if path.exists():
        return np.load(path, allow_pickle=False)
    if artifact.latent_data:
        write_file_if_missing(path, artifact.latent_data)
        return array_from_npy_bytes(artifact.latent_data)
    raise FileNotFoundError(f"Latent artifact is unavailable: {artifact.image_id}")


def image_bytes(artifact: LatentImage) -> bytes:
    path = Path(artifact.image_path)
    if path.exists():
        return path.read_bytes()
    if artifact.image_data:
        write_file_if_missing(path, artifact.image_data)
        return artifact.image_data
    raise FileNotFoundError(f"Image artifact is unavailable: {artifact.image_id}")
