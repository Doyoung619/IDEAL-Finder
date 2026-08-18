import torch
from PIL import Image

from core.demographic_classifier import AGE_LABELS, FairFaceDemographicClassifier


class FixedFairFace(torch.nn.Module):
    def forward(self, values):
        logits = torch.zeros((len(values), 18), device=values.device)
        logits[:, 12] = 3.0
        logits[:, 13] = 2.0
        return logits


def test_fairface_returns_age_probabilities_in_documented_order(tmp_path):
    classifier = FairFaceDemographicClassifier(tmp_path / "unused.pth", device="cpu")
    classifier._model = FixedFairFace()
    result = classifier.predict_proba([Image.new("RGB", (32, 32), "white")])

    assert AGE_LABELS[3:5] == ("20_29", "30_39")
    assert result["age"].shape == (1, 9)
    assert torch.allclose(result["age"].sum(dim=1), torch.ones(1))
