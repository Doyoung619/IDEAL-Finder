from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from core.pca_utils import LatentProjector


class PairwisePreferenceModel:
    def __init__(self, projector: LatentProjector) -> None:
        self.projector = projector
        self.model: LogisticRegression | None = None

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def fit(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> "PairwisePreferenceModel":
        if not pairs:
            raise ValueError("At least one winner/loser pair is required")

        winners = np.vstack([winner for winner, _ in pairs])
        losers = np.vstack([loser for _, loser in pairs])
        winner_features = self.projector.transform(winners)
        loser_features = self.projector.transform(losers)
        differences = winner_features - loser_features
        features = np.vstack([differences, -differences])
        labels = np.concatenate(
            [np.ones(len(differences)), np.zeros(len(differences))]
        )
        self.model = LogisticRegression(
            C=1.0,
            fit_intercept=False,
            max_iter=2000,
            random_state=0,
        )
        self.model.fit(features, labels)
        return self

    def score(self, latents: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Preference model is not fitted")
        features = self.projector.transform(latents)
        return (features @ self.model.coef_[0]).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Preference model is not fitted")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, destination)

    def load(self, path: str | Path) -> "PairwisePreferenceModel":
        self.model = joblib.load(path)
        return self

