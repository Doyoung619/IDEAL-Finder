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
    """Expose only the fixed entropy-based synthetic query algorithm."""
    if config.query.algorithm != "entropy":
        raise ValueError("Only the entropy query algorithm is currently supported.")
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
    )


def catalog_by_key(config) -> dict[str, AlgorithmSpec]:
    return {item.key: item for item in algorithm_catalog(config)}


def parse_parameters(
    form: Mapping[str, object],
    algorithm: AlgorithmSpec,
) -> dict[str, float | int | bool]:
    del form
    return {parameter.key: parameter.default for parameter in algorithm.parameters}
