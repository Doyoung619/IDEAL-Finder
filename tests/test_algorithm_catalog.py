import pytest

from app.settings import load_config
from core.algorithm_catalog import algorithm_catalog, catalog_by_key, parse_parameters


def test_catalog_exposes_entropy_and_rc_mlq():
    config = load_config(demo_override=True)
    catalog = algorithm_catalog(config)
    assert [item.key for item in catalog] == ["entropy", "rc_mlq"]
    assert catalog[0].label == "Entropy Query"
    assert catalog_by_key(config)["entropy"].parameters == ()
    assert parse_parameters({}, catalog[0]) == {}


def test_catalog_is_independent_of_legacy_default_algorithm_field():
    config = load_config(demo_override=True)
    config.query._values["algorithm"] = "unsupported"
    assert [item.key for item in algorithm_catalog(config)] == ["entropy", "rc_mlq"]
