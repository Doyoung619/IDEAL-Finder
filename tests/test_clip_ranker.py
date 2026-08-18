from __future__ import annotations

import numpy as np
import pytest
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


def test_persona_embedding_api_is_normalized_mock_and_fails_closed_in_research():
    image = Image.new("RGB", (16, 16), color=(120, 100, 90))
    mock = CLIPRanker(enabled=False, device="cpu", allow_mock=True)

    image_embedding = mock.encode_images([image])
    text_embedding = mock.encode_texts(["East Asian adult portrait"])
    assert image_embedding.dtype == np.float32
    assert text_embedding.dtype == np.float32
    assert np.linalg.norm(image_embedding[0]) == pytest.approx(1.0)
    assert np.linalg.norm(text_embedding[0]) == pytest.approx(1.0)
    assert mock.backend == "mock_deterministic"

    research = CLIPRanker(enabled=False, device="cpu", require_real=True)
    with pytest.raises(RuntimeError, match="requires a real OpenCLIP"):
        research.encode_texts(["portrait"])
