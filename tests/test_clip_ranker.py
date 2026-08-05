from __future__ import annotations

import numpy as np
from PIL import Image

from core.clip_ranker import CLIPRanker


def test_clip_fallback_is_deterministic_and_prompt_conditioned():
    ranker = CLIPRanker(enabled=False, device="cpu")
    images = [Image.new("RGB", (16, 16), color=(120, 100, 90)) for _ in range(3)]
    features = np.arange(12, dtype=np.float32).reshape(3, 4)

    first = ranker.score(images, "elegant adult portrait", features)
    second = ranker.score(images, "elegant adult portrait", features)
    other = ranker.score(images, "sporty adult portrait", features)

    assert first.shape == (3,)
    assert np.allclose(first, second)
    assert not np.allclose(first, other)
    assert ranker.backend == "latent_hash_fallback"

