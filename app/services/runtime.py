from __future__ import annotations

from pathlib import Path

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
        self.query_strategy = None

    def ensure_ready(self) -> None:
        self.projector.ensure_fitted(
            generator=self.generator,
            sample_count=self.config.search.pca_fit_samples,
            seed=self.config.experiment.seed,
        )
        if self.query_strategy is None:
            self.query_strategy = create_query_strategy(
                self.config,
                self.generator,
                self.projector,
            )

    def strategy_for(self, mode: str, parameters: dict):
        self.ensure_ready()
        return create_query_strategy(
            self.config,
            self.generator,
            self.projector,
            mode=mode,
            parameters=parameters,
        )
