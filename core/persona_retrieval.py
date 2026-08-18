from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from core.persona_pool import PersonaPool


@dataclass(frozen=True)
class PersonaRetrievalConfig:
    candidate_count: int = 8
    shortlist_size: int = 64
    mmr_lambda: float = 0.75
    max_pages: int = 3
    base_weight: float = 0.25
    full_weight: float = 0.40
    categories_weight: float = 0.35
    quality_tiebreaker: float = 0.01

    def __post_init__(self) -> None:
        if self.candidate_count < 1 or self.shortlist_size < self.candidate_count:
            raise ValueError("shortlist_size must be at least candidate_count")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if not np.isclose(
            self.base_weight + self.full_weight + self.categories_weight, 1.0
        ):
            raise ValueError("persona semantic weights must sum to 1")


@dataclass(frozen=True)
class RetrievedPersonaCandidate:
    pool_id: str
    pool_index: int
    semantic_score: float
    mmr_score: float
    semantic_rank: int


class PersonaRetriever:
    def __init__(self, clip_ranker, config: PersonaRetrievalConfig) -> None:
        self.clip_ranker = clip_ranker
        self.config = config

    def retrieve(
        self,
        profile,
        pool: PersonaPool,
        excluded_pool_ids: set[str] | None = None,
        seed: int = 0,
    ) -> list[RetrievedPersonaCandidate]:
        excluded = {str(value) for value in (excluded_pool_ids or set())}
        texts = [profile.base_prompt, profile.full_prompt, *profile.category_prompts]
        text_embeddings = self.clip_ranker.encode_texts(texts)
        image_embeddings = np.asarray(pool.clip_image_embedding, dtype=np.float32)
        if text_embeddings.shape[1] != image_embeddings.shape[1]:
            raise ValueError("Persona pool and text embedding dimensions do not match")
        base_scores = image_embeddings @ text_embeddings[0]
        full_scores = image_embeddings @ text_embeddings[1]
        if profile.category_prompts:
            category_embeddings = text_embeddings[2:]
            category_weights = np.asarray(profile.category_weights, dtype=np.float32)
            if len(category_weights) != len(category_embeddings) or not np.isclose(category_weights.sum(), 1.0):
                raise ValueError("Persona category prompt weights must be normalized")
            category_scores = (image_embeddings @ category_embeddings.T) @ category_weights
        else:
            category_scores = base_scores
        semantic = (
            self.config.base_weight * base_scores
            + self.config.full_weight * full_scores
            + self.config.categories_weight * category_scores
        )

        eligible = [
            index
            for index, pool_id in enumerate(pool.pool_ids)
            if str(pool_id) not in excluded
        ]
        if len(eligible) < self.config.candidate_count:
            raise RuntimeError("Not enough unseen persona candidates remain in the pool")
        ordered = sorted(
            eligible,
            key=lambda index: (
                -float(semantic[index]),
                -self.config.quality_tiebreaker * float(pool.quality_score[index]),
                self._tie_value(str(pool.pool_ids[index]), seed),
            ),
        )
        rank_by_index = {index: rank + 1 for rank, index in enumerate(ordered)}
        shortlist = ordered[: min(self.config.shortlist_size, len(ordered))]
        selected: list[int] = []
        mmr_values: dict[int, float] = {}
        while len(selected) < self.config.candidate_count:
            best_index = None
            best_key = None
            best_mmr = float("-inf")
            for index in shortlist:
                if index in selected:
                    continue
                redundancy = (
                    max(float(image_embeddings[index] @ image_embeddings[item]) for item in selected)
                    if selected
                    else 0.0
                )
                mmr = (
                    self.config.mmr_lambda * float(semantic[index])
                    - (1.0 - self.config.mmr_lambda) * redundancy
                )
                key = (
                    mmr,
                    float(semantic[index]),
                    self.config.quality_tiebreaker * float(pool.quality_score[index]),
                    -self._tie_value(str(pool.pool_ids[index]), seed),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
                    best_mmr = mmr
            if best_index is None:
                raise RuntimeError("Persona MMR selection exhausted its shortlist")
            selected.append(best_index)
            mmr_values[best_index] = best_mmr
        return [
            RetrievedPersonaCandidate(
                pool_id=str(pool.pool_ids[index]),
                pool_index=int(index),
                semantic_score=float(semantic[index]),
                mmr_score=float(mmr_values[index]),
                semantic_rank=int(rank_by_index[index]),
            )
            for index in selected
        ]

    @staticmethod
    def _tie_value(pool_id: str, seed: int) -> int:
        payload = f"{int(seed)}:{pool_id}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
