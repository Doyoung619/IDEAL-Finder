from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def latent_fingerprint(latent: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(latent.astype(np.float32))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:20]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def relative_web_path(path: str | Path, project_root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()

