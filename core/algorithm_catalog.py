from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


ParameterKind = Literal["float", "int", "bool"]


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    description: str
    kind: ParameterKind
    default: float | int | bool
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    category: str
    summary: str
    parameters: tuple[ParameterSpec, ...]
    speed_tier: str = "standard"
    speed_label: str = ""


def algorithm_catalog(config) -> tuple[AlgorithmSpec, ...]:
    """Expose the two pre-registered algorithms used by the fixed study schedule."""
    return (
        AlgorithmSpec(
            key="entropy",
            label="Entropy Query",
            category="Bayesian information-based query synthesis",
            summary=(
                "현재 posterior와 거리 기반 M-way 선택 모델에서 mutual information을 "
                "최대화하도록 latent 후보를 연속 최적화해 직접 생성합니다."
            ),
            parameters=(),
            speed_tier="optimized",
            speed_label="latent 직접 최적화",
        ),
        AlgorithmSpec(
            key="rc_mlq",
            label="RC-MLQ (Ours)",
            category="Resolution-calibrated Bayesian line query",
            summary=(
                "posterior 최대분산 방향의 quantile 배치를 만들고, posterior 폭과 독립된 "
                "물리적 resolution grid에서 expected information gain을 최대화합니다."
            ),
            parameters=(),
            speed_tier="optimized",
            speed_label="고정 물리 해상도 EIG 보정",
        ),
    )


def catalog_by_key(config) -> dict[str, AlgorithmSpec]:
    return {item.key: item for item in algorithm_catalog(config)}


def parse_parameters(
    form: Mapping[str, object],
    algorithm: AlgorithmSpec,
) -> dict[str, float | int | bool]:
    del form
    return {parameter.key: parameter.default for parameter in algorithm.parameters}
