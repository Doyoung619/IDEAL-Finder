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
from core.conditional_prior import ConditionalPCAPrior
from core.demographic_classifier import FairFaceDemographicClassifier


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
        self._conditional_priors = {}
        self._demographic_classifiers = {}

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
        conditional_prior = None
        demographic_classifier = None
        if mode == "demographic_constrained_mvlq":
            configured = {
                "gender_threshold": self.config.demographic.gender_threshold,
                "race_threshold": self.config.demographic.race_threshold,
                "backtrack_factor": self.config.query.backtrack_factor,
                "min_alpha": self.config.query.min_alpha,
                "max_backtracking_steps": (
                    self.config.query.max_backtracking_steps
                ),
                "center_candidates": self.config.query.center_candidates,
                "beta": self.config.preference.beta,
                "distance_space": self.config.preference.distance_space,
                "seed": self.config.conditional_prior.seed,
            }
            configured.update(parameters)
            parameters = configured
            conditional_prior, demographic_classifier = (
                self.conditional_components(parameters)
            )
        return create_query_strategy(
            self.config,
            self.generator,
            self.projector,
            mode=mode,
            parameters=parameters,
            conditional_prior=conditional_prior,
            demographic_classifier=demographic_classifier,
        )

    def conditional_components(self, parameters: dict | None = None):
        """Load the configured prior and FairFace adapter on first constrained use."""
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
        weights = str(
            values.get(
                "fairface_weights", self.config.demographic.weights
            )
        )
        if self.config.demographic.classifier != "fairface":
            raise ValueError("Only the 'fairface' demographic classifier is supported")
        if weights not in self._demographic_classifiers:
            self._demographic_classifiers[weights] = FairFaceDemographicClassifier(
                weights_path=weights,
                device=self.config.demographic.device,
                batch_size=self.config.demographic.batch_size,
            )
        return (
            self._conditional_priors[prior_path],
            self._demographic_classifiers[weights],
        )
