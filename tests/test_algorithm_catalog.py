from __future__ import annotations

from app.settings import load_config
from core.algorithm_catalog import algorithm_catalog, catalog_by_key, parse_parameters


def test_catalog_exposes_only_mvlq():
    config = load_config(demo_override=True)

    assert [algorithm.key for algorithm in algorithm_catalog(config)] == ["mvlq"]
    assert catalog_by_key(config)["mvlq"].parameters == ()


def test_mvlq_has_no_manual_hyperparameters():
    config = load_config(demo_override=True)
    algorithm = catalog_by_key(config)["mvlq"]

    assert parse_parameters({}, algorithm) == {}
