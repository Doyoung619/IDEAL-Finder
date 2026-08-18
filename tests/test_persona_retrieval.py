import numpy as np
from PIL import Image

from app.services.persona_service import PERSONA_CATEGORIES, build_persona_prompt_bundle
from core.persona_retrieval import PersonaRetrievalConfig, PersonaRetriever
from core.persona_pool import PersonaPool


def make_pool(root, size=3):
    (root / "images").mkdir(parents=True)
    paths = []
    for index in range(size):
        path = f"images/{index:06d}.png"
        Image.new("RGB", (16, 16), (index * 40, 80, 120)).save(root / path)
        paths.append(path)
    return PersonaPool(
        root=root,
        pool_ids=np.asarray([f"female-v1-{index:06d}" for index in range(size)]),
        theta=np.zeros((size, 2), dtype=np.float32),
        w=np.zeros((size, 4), dtype=np.float32),
        clip_image_embedding=np.eye(size, dtype=np.float32),
        quality_score=np.full(size, 0.8, dtype=np.float32),
        gender_probability=np.full(size, 0.9, dtype=np.float32),
        age_20_29_probability=np.full(size, 0.9, dtype=np.float32),
        generator_seed=np.arange(size),
        image_paths=tuple(paths),
        metadata={"gender": "female"},
    )


class FixedTextEncoder:
    backend = "test"

    def encode_texts(self, texts):
        values = np.zeros((len(texts), 3), dtype=np.float32)
        values[:, 0] = 1.0
        return values


def test_persona_retrieval_excludes_seen_ids_and_returns_exact_diverse_count(tmp_path):
    pool = make_pool(tmp_path)
    answers = {category.key: ["no_preference"] for category in PERSONA_CATEGORIES}
    profile = build_persona_prompt_bundle("female", answers, [])
    retriever = PersonaRetriever(
        FixedTextEncoder(),
        PersonaRetrievalConfig(candidate_count=2, shortlist_size=3),
    )
    result = retriever.retrieve(
        profile, pool, excluded_pool_ids={"female-v1-000000"}, seed=7
    )

    assert len(result) == 2
    assert {item.pool_id for item in result} == {
        "female-v1-000001",
        "female-v1-000002",
    }
    assert all(np.isfinite(item.mmr_score) for item in result)
