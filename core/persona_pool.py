from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


POOL_FORMAT_VERSION = "persona_pool_v1"
REQUIRED_ARRAYS = (
    "pool_ids",
    "theta",
    "w",
    "clip_image_embedding",
    "quality_score",
    "gender_probability",
    "age_20_29_probability",
    "generator_seed",
)


@dataclass
class PersonaPool:
    root: Path
    pool_ids: np.ndarray
    theta: np.ndarray
    w: np.ndarray
    clip_image_embedding: np.ndarray
    quality_score: np.ndarray
    gender_probability: np.ndarray
    age_20_29_probability: np.ndarray
    generator_seed: np.ndarray
    image_paths: tuple[str, ...]
    metadata: dict

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        size = len(self.pool_ids)
        arrays = {
            name: np.asarray(getattr(self, name))
            for name in REQUIRED_ARRAYS
            if name != "pool_ids"
        }
        if size == 0:
            raise ValueError("Persona pool must contain at least one candidate")
        if len(self.image_paths) != size or any(len(value) != size for value in arrays.values()):
            raise ValueError("Persona pool arrays and image paths must have equal length")
        if self.theta.ndim != 2 or self.w.ndim != 2 or self.clip_image_embedding.ndim != 2:
            raise ValueError("theta, w, and CLIP embeddings must be 2D arrays")
        numeric = [self.theta, self.w, self.clip_image_embedding]
        numeric.extend(arrays[name] for name in REQUIRED_ARRAYS[4:])
        if any(not np.isfinite(value).all() for value in numeric):
            raise ValueError("Persona pool contains NaN or Inf")
        norms = np.linalg.norm(self.clip_image_embedding, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Persona pool CLIP embeddings must be L2-normalized")
        if len(set(map(str, self.pool_ids))) != size:
            raise ValueError("Persona pool IDs must be unique")

    @property
    def size(self) -> int:
        return len(self.pool_ids)

    @property
    def theta_dimension(self) -> int:
        return int(self.theta.shape[1])

    def image_path(self, index: int) -> Path:
        return self.root / self.image_paths[int(index)]

    def load_image(self, index: int) -> Image.Image:
        with Image.open(self.image_path(index)) as image:
            return image.convert("RGB")

    def validate_thresholds(
        self,
        gender_threshold: float,
        age_20_29_threshold: float,
        minimum_quality: float,
    ) -> None:
        checks = {
            "gender": (self.gender_probability, gender_threshold),
            "age 20-29": (self.age_20_29_probability, age_20_29_threshold),
            "quality": (self.quality_score, minimum_quality),
        }
        for label, (values, threshold) in checks.items():
            failures = int(np.count_nonzero(np.asarray(values) < float(threshold)))
            if failures:
                raise ValueError(
                    f"Persona pool has {failures} candidates below the {label} threshold"
                )

    @classmethod
    def load(cls, root: str | Path, minimum_size: int = 1) -> "PersonaPool":
        directory = Path(root)
        pool_path = directory / "pool.npz"
        if not pool_path.exists():
            raise FileNotFoundError(f"Persona pool not found: {pool_path}")
        with np.load(pool_path, allow_pickle=False) as archive:
            missing = set(REQUIRED_ARRAYS) - set(archive.files)
            if missing:
                raise ValueError(f"Persona pool is missing arrays: {sorted(missing)}")
            values = {name: np.asarray(archive[name]) for name in REQUIRED_ARRAYS}
        metadata_path = directory / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        manifest_path = directory / "manifest.jsonl"
        image_paths: dict[int, str] = {}
        if manifest_path.exists():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    image_paths[int(item["pool_index"])] = str(item["image_path"])
        ordered_paths = tuple(
            image_paths.get(index, f"images/{index:06d}.png")
            for index in range(len(values["pool_ids"]))
        )
        pool = cls(
            root=directory,
            image_paths=ordered_paths,
            metadata=metadata,
            **values,
        )
        if pool.size < int(minimum_size):
            raise RuntimeError(
                f"Persona pool has {pool.size} candidates; at least {minimum_size} are required"
            )
        for relative in pool.image_paths:
            if not (directory / relative).exists():
                raise FileNotFoundError(f"Persona pool image not found: {directory / relative}")
        return pool

    def save(self, root: str | Path | None = None) -> None:
        directory = Path(root) if root is not None else self.root
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "pool.npz.tmp"
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                pool_ids=np.asarray(self.pool_ids, dtype=str),
                theta=np.asarray(self.theta, dtype=np.float32),
                w=np.asarray(self.w, dtype=np.float32),
                clip_image_embedding=np.asarray(self.clip_image_embedding, dtype=np.float32),
                quality_score=np.asarray(self.quality_score, dtype=np.float32),
                gender_probability=np.asarray(self.gender_probability, dtype=np.float32),
                age_20_29_probability=np.asarray(self.age_20_29_probability, dtype=np.float32),
                generator_seed=np.asarray(self.generator_seed, dtype=np.int64),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory / "pool.npz")

        metadata = {"format_version": POOL_FORMAT_VERSION, "size": self.size, **self.metadata}
        _atomic_text(directory / "metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
        manifest_lines = []
        for index, pool_id in enumerate(self.pool_ids):
            manifest_lines.append(
                json.dumps(
                    {
                        "pool_id": str(pool_id),
                        "pool_index": index,
                        "image_path": self.image_paths[index],
                        "gender": metadata.get("gender"),
                        "gender_probability": float(self.gender_probability[index]),
                        "age_20_29_probability": float(self.age_20_29_probability[index]),
                        "quality_score": float(self.quality_score[index]),
                        "seed": int(self.generator_seed[index]),
                    },
                    ensure_ascii=False,
                )
            )
        _atomic_text(directory / "manifest.jsonl", "\n".join(manifest_lines) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
