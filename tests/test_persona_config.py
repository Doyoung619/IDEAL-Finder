from app.settings import PROJECT_ROOT, load_config


def test_persona_study_config_is_separate_and_resolves_pool_paths():
    config = load_config(PROJECT_ROOT / "configs" / "persona_study.yaml")

    assert config.experiment.m_list == [8, 4, 2]
    assert config.experiment.algorithm_order == ["entropy", "rc_mlq"]
    assert config.experiment.answer_key_faces == 0
    assert config.experiment.recommendation_conditions == []
    assert config.persona.required
    assert config.persona.fixed_age_range == "20_29"
    assert config.persona.fixed_race == "unrestricted"
    assert config.persona.female_pool.startswith(str(PROJECT_ROOT))
    assert config.generator.mode == "stylegan3"
    assert config.conditional_prior.artifact_path.endswith(
        "{gender}_20_29_d12.npz"
    )
