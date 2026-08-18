from pathlib import Path

import pytest

from app.settings import load_config
from core.generator import StyleGAN2ADAGenerator, StyleGAN3Generator


def test_missing_environment_variable_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_STYLEGAN_TEST_PATH", raising=False)
    config_path = tmp_path / "conditional.yaml"
    config_path.write_text(
        'generator:\n  network_path: "${MISSING_STYLEGAN_TEST_PATH}"\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="MISSING_STYLEGAN_TEST_PATH"):
        load_config(config_path)


def test_stylegan_adapter_is_lazy_without_checkpoint(tmp_path):
    generator = StyleGAN2ADAGenerator(
        repo_path=str(tmp_path / "missing_repo"),
        network_path=str(tmp_path / "missing.pkl"),
        device="cpu",
    )

    assert generator.generator_name.startswith("stylegan2_ada_")
    assert generator._network is None
    assert not Path(generator.network_path).exists()


def test_stylegan3_adapter_is_lazy_without_checkpoint(tmp_path):
    generator = StyleGAN3Generator(
        repo_path=str(tmp_path / "missing_repo"),
        network_path=str(tmp_path / "missing.pkl"),
        device="cpu",
    )

    assert generator.generator_name.startswith("stylegan3_")
    assert generator._network is None
    assert not Path(generator.network_path).exists()


def test_default_m_conditions_exclude_16():
    config = load_config(demo_override=False)
    assert list(config.experiment.m_list) == [8, 4, 2]
    assert list(config.experiment.algorithm_order) == ["entropy", "rc_mlq"]
