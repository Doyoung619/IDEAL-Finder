import pytest

from app.settings import load_config
from core.algorithm_catalog import algorithm_catalog, catalog_by_key, parse_parameters


def test_catalog_exposes_only_entropy():
    config = load_config(demo_override=True)
    catalog = algorithm_catalog(config)
    assert [item.key for item in catalog] == ["entropy"]
    assert catalog[0].label == "Entropy Query"
    assert catalog_by_key(config)["entropy"].parameters == ()
    assert parse_parameters({}, catalog[0]) == {}


def test_catalog_rejects_any_other_algorithm():
    config = load_config(demo_override=True)
    config.query._values["algorithm"] = "unsupported"
    with pytest.raises(
        ValueError, match="Only the entropy query algorithm is currently supported"
    ):
        algorithm_catalog(config)
