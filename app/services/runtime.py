from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

from core.adult_control import AdultFaceController
from core.age_appearance_control import AgeAppearanceController
from core.clip_ranker import CLIPRanker
from core.face_filter import FaceQualityFilter
from core.face_region_control import FaceRegionController
from core.gender_control import GenderController
from core.generator import create_generator
from core.pca_utils import LatentProjector
from core.portrait_control import PortraitQualityController
from core.query_strategy import create_query_strategy
from core.conditional_prior import ConditionalPCAPrior, DemographicCondition
from core.persona_pool import PersonaPool
from core.persona_retrieval import PersonaRetrievalConfig, PersonaRetriever


class ExperimentRuntime:
    def __init__(self, config) -> None:
        self.config = config
        self.generator = create_generator(config)
        projector_name = (
            f"latent_pca_{self.generator.generator_name}_"
            f"{config.search.pca_dimensions}d.joblib"
        )
        self.projector = LatentProjector(
            Path(config.paths.data_dir) / "models" / projector_name,
            dimensions=config.search.pca_dimensions,
        )
        self.clip_ranker = CLIPRanker(
            enabled=config.clip.enabled,
            model_name=config.clip.model_name,
            pretrained=config.clip.pretrained,
            device=config.clip.device,
            batch_size=config.clip.batch_size,
            require_real=(
                bool(config.persona.require_real_clip)
                and config.generator.mode != "demo"
            ),
            allow_mock=config.generator.mode == "demo",
        )
        self.gender_controller = GenderController(
            clip_ranker=self.clip_ranker,
            confidence_threshold=config.filters.gender_confidence_threshold,
            strict=config.filters.strict_gender,
        )
        self.adult_controller = AdultFaceController(
            clip_ranker=self.clip_ranker,
            confidence_threshold=config.filters.adult_confidence_threshold,
        )
        self.portrait_controller = PortraitQualityController(
            clip_ranker=self.clip_ranker,
            confidence_threshold=config.filters.portrait_confidence_threshold,
        )
        self.face_region_controller = FaceRegionController(self.clip_ranker)
        self.age_appearance_controller = AgeAppearanceController(self.clip_ranker)
        self.quality_filter = FaceQualityFilter(
            minimum_quality=config.filters.minimum_quality
        )
        self._conditional_priors = {}
        self._persona_pools = {}
        # One process should own one GPU worker. This lock prevents overlapping
        # StyleGAN/OpenCLIP batches from exhausting GPU memory.
        self.generation_lock = threading.RLock()

    def ensure_ready(self) -> None:
        self.projector.ensure_fitted(
            generator=self.generator,
            sample_count=self.config.search.pca_fit_samples,
            seed=self.config.experiment.seed,
        )

    def strategy_for(self, mode: str, parameters: dict):
        allowed = set(self.config.experiment.algorithm_order)
        if mode not in allowed:
            raise ValueError(f"Unsupported query algorithm: {mode}")
        conditional_prior = self.conditional_prior(parameters)
        return create_query_strategy(
            self.config,
            self.generator,
            self.projector,
            mode=mode,
            parameters=parameters,
            conditional_prior=conditional_prior,
        )

    def conditional_prior(
        self, parameters: dict | None = None
    ) -> ConditionalPCAPrior:
        """Load the requested condition prior or create the deterministic demo prior."""
        values = parameters or {}
        condition_gender = str(
            values.get("condition_gender", self.config.demographic.gender)
        )
        condition_races = tuple(
            values.get(
                "condition_races", self.config.demographic.race_targets
            )
        )
        prior_template = str(
            values.get(
                "prior_path", self.config.conditional_prior.artifact_path
            )
        )
        prior_path = prior_template.format(
            gender=condition_gender,
            race="_".join(condition_races),
        )
        if prior_path not in self._conditional_priors:
            if self.config.generator.mode == "demo":
                dimension = min(
                    int(self.config.conditional_prior.dimension),
                    int(self.generator.latent_dim),
                )
                prior = ConditionalPCAPrior(
                    condition=DemographicCondition(
                        condition_gender, condition_races
                    ),
                    mu_w=np.zeros(self.generator.latent_dim),
                    components=np.eye(self.generator.latent_dim)[:dimension],
                    eigenvalues=np.ones(dimension),
                    explained_variance_ratio=np.full(
                        dimension, 1.0 / dimension
                    ),
                    accepted_samples=1,
                    seed=int(self.config.query.seed),
                )
            else:
                prior = ConditionalPCAPrior.load(prior_path)
            if prior.w_dimension != self.generator.latent_dim:
                raise ValueError(
                    "Conditional prior W dimension does not match the generator"
                )
            if prior.condition.gender != condition_gender or tuple(
                prior.condition.race_targets
            ) != condition_races:
                raise ValueError(
                    "Conditional prior condition does not match the requested "
                    f"gender/races: {condition_gender}/{condition_races}"
                )
            self._conditional_priors[prior_path] = prior
        return self._conditional_priors[prior_path]

    def persona_pool(self, gender: str) -> PersonaPool:
        if gender not in {"female", "male"}:
            raise ValueError("Persona pool gender must be female or male")
        if gender in self._persona_pools:
            return self._persona_pools[gender]
        configured = (
            self.config.persona.female_pool
            if gender == "female"
            else self.config.persona.male_pool
        )
        if self.config.generator.mode == "demo":
            root = Path(self.config.paths.cache_dir) / "demo_persona_pools" / gender
            if not (root / "pool.npz").exists():
                self._build_demo_persona_pool(root, gender)
            minimum_size = int(self.config.persona_pool.minimum_usable_size)
        else:
            root = Path(configured)
            minimum_size = int(self.config.persona_pool.minimum_usable_size)
        pool = PersonaPool.load(root, minimum_size=minimum_size)
        if self.config.generator.mode != "demo":
            pool.validate_thresholds(
                gender_threshold=float(self.config.persona_pool.gender_threshold),
                age_20_29_threshold=float(self.config.persona_pool.age_20_29_threshold),
                minimum_quality=float(self.config.persona_pool.minimum_quality),
            )
        condition_races = list(self.config.demographic.race_targets)
        if pool.theta_dimension != self.conditional_prior(
            {"condition_gender": gender, "condition_races": condition_races}
        ).dimension:
            raise ValueError("Persona pool theta dimension does not match its prior")
        if pool.metadata.get("gender") not in {None, gender}:
            raise ValueError("Persona pool gender metadata does not match")
        expected_population = str(self.config.persona.fixed_race)
        if pool.metadata.get("fixed_race") not in {None, expected_population}:
            raise ValueError("Persona pool population metadata does not match")
        self._persona_pools[gender] = pool
        return pool

    def persona_retriever(self) -> PersonaRetriever:
        weights = self.config.persona.semantic_weights
        return PersonaRetriever(
            self.clip_ranker,
            PersonaRetrievalConfig(
                candidate_count=int(self.config.persona.candidate_count),
                shortlist_size=int(self.config.persona.shortlist_size),
                mmr_lambda=float(self.config.persona.mmr_lambda),
                max_pages=int(self.config.persona.max_candidate_pages),
                base_weight=float(weights.base),
                full_weight=float(weights.full),
                categories_weight=float(weights.categories),
            ),
        )

    def _build_demo_persona_pool(self, root: Path, gender: str) -> None:
        condition_races = list(self.config.demographic.race_targets)
        prior = self.conditional_prior(
            {"condition_gender": gender, "condition_races": condition_races}
        )
        size = max(
            int(self.config.persona_pool.target_size),
            int(self.config.persona.candidate_count)
            * int(self.config.persona.max_candidate_pages),
        )
        gender_offset = 0 if gender == "female" else 100_000
        seed = int(self.config.experiment.seed) + gender_offset
        random = np.random.default_rng(seed)
        theta = prior.sample_theta(size, generator=random).astype(np.float32)
        w = np.asarray(prior.theta_to_w(theta), dtype=np.float32)
        images = self.generator.decode(w)
        root.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for index, image in enumerate(images):
            relative = f"images/{index:06d}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format="PNG", optimize=True)
            image_paths.append(relative)
        embeddings = self.clip_ranker.encode_images(images)
        pool = PersonaPool(
            root=root,
            pool_ids=np.asarray(
                [f"{gender}-demo-v1-{index:06d}" for index in range(size)]
            ),
            theta=theta,
            w=w,
            clip_image_embedding=embeddings,
            quality_score=np.full(size, 1.0, dtype=np.float32),
            gender_probability=np.full(size, 1.0, dtype=np.float32),
            age_20_29_probability=np.full(size, 1.0, dtype=np.float32),
            generator_seed=np.arange(seed, seed + size, dtype=np.int64),
            image_paths=tuple(image_paths),
            metadata={
                "pool_version": f"{gender}-demo-v1",
                "gender": gender,
                "fixed_race": str(self.config.persona.fixed_race),
                "fixed_age": "20_29",
                "clip_backend": self.clip_ranker.backend,
                "scientific_result": False,
            },
        )
        pool.save()
