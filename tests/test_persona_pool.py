import numpy as np
import pytest
from PIL import Image

from core.persona_pool import PersonaPool


def make_pool(root, size=3):
    (root / "images").mkdir(parents=True)
    paths = []
    for index in range(size):
        path = f"images/{index:06d}.png"
        Image.new("RGB", (16, 16), (index * 40, 80, 120)).save(root / path)
        paths.append(path)
    embeddings = np.eye(size, dtype=np.float32)
    return PersonaPool(
        root=root,
        pool_ids=np.asarray([f"female-v1-{index:06d}" for index in range(size)]),
        theta=np.arange(size * 2, dtype=np.float32).reshape(size, 2),
        w=np.arange(size * 4, dtype=np.float32).reshape(size, 4),
        clip_image_embedding=embeddings,
        quality_score=np.full(size, 0.8, dtype=np.float32),
        gender_probability=np.full(size, 0.9, dtype=np.float32),
        age_20_29_probability=np.full(size, 0.9, dtype=np.float32),
        generator_seed=np.arange(size),
        image_paths=tuple(paths),
        metadata={"gender": "female"},
    )


def test_persona_pool_round_trip_uses_non_pickle_npz(tmp_path):
    pool = make_pool(tmp_path)
    pool.save()
    loaded = PersonaPool.load(tmp_path, minimum_size=3)

    assert loaded.size == 3
    assert loaded.theta.dtype == np.float32
    assert loaded.pool_ids.tolist() == pool.pool_ids.tolist()
    assert loaded.load_image(1).size == (16, 16)


def test_persona_pool_threshold_validation_fails_closed(tmp_path):
    pool = make_pool(tmp_path)
    pool.gender_probability[1] = 0.2
    with pytest.raises(ValueError, match="gender threshold"):
        pool.validate_thresholds(0.8, 0.55, 0.18)
