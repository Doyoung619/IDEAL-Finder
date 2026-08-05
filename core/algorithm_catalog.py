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
    del config
    return (
        AlgorithmSpec(
            key="mvlq",
            label="Maximum-Variance Line Query",
            category="Bayesian active preference estimation",
            summary=(
                "현재 posterior가 가장 넓게 퍼진 방향을 찾아 MAP 중심의 직선 위에 "
                "M개 얼굴을 균등 배치하고, 선택 결과로 posterior를 갱신합니다."
            ),
            parameters=(),
            speed_tier="fastest",
            speed_label="단일 고정 알고리즘",
        ),
    )


def catalog_by_key(config) -> dict[str, AlgorithmSpec]:
    return {algorithm.key: algorithm for algorithm in algorithm_catalog(config)}


def parse_parameters(
    form: Mapping[str, object],
    algorithm: AlgorithmSpec,
) -> dict[str, float | int | bool]:
    del form
    return {parameter.key: parameter.default for parameter in algorithm.parameters}
