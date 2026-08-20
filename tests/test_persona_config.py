import hashlib
import json

from app.settings import PROJECT_ROOT, load_config


FINAL_CONFIG_SHA256 = (
    "2a225eb9355af0f4198088e0340329e31bd0ec1fbbdb90f53631900634101b71"
)

POST_AUDIT_PRODUCTION_CHANGES = {
    "app/services/generation_service.py",
    "configs/persona_study.yaml",
    "core/query_diversity.py",
    "core/query_strategy.py",
}


def test_persona_study_config_is_separate_and_resolves_pool_paths():
    config_path = PROJECT_ROOT / "configs" / "persona_study.yaml"
    config = load_config(config_path)

    assert config.experiment.m_list == [8, 4, 2]
    assert config.experiment.algorithm_order == ["entropy", "rc_mlq"]
    assert config.experiment.answer_key_faces == 0
    assert config.experiment.rounds_per_m == 8
    assert config.experiment.recommendation_conditions == []
    assert config.persona.required
    assert config.persona.max_candidate_pages == 2
    assert config.persona.fixed_age_range == "20_29"
    assert config.persona.fixed_race == "east_asian"
    assert config.demographic.race_targets == ["east_asian"]
    assert config.query_diversity.enabled
    assert config.query_diversity.image_similarity_threshold is None
    assert config.query_diversity.latent_min_distance == 0.75
    assert config.query.exploration_rho == 0.15
    assert config.query.novelty_candidate_fraction == 0.50
    assert config.query.novelty_candidate_pool_size == 64
    assert config.query.posterior_mc_samples == 256
    assert config.query.num_restarts == 3
    assert config.query.optimization_steps == 60
    assert config.query.rc_posterior_samples == 2048
    assert config.query.rc_resolution_steps == 80
    assert not config.query.adaptive_beta_enabled
    assert config.persona.female_pool.startswith(str(PROJECT_ROOT))
    assert config.generator.mode == "stylegan3"
    assert config.conditional_prior.artifact_path.endswith(
        "{gender}_20_29_{race}_d12.npz"
    )
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == FINAL_CONFIG_SHA256


def test_unmodified_audited_research_files_still_match_baseline():
    manifest = json.loads(
        (PROJECT_ROOT / "docs" / "algorithm_baseline_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for relative_path, expected_hash in manifest["files"].items():
        if relative_path in POST_AUDIT_PRODUCTION_CHANGES:
            continue
        actual_hash = hashlib.sha256(
            (PROJECT_ROOT / relative_path).read_bytes()
        ).hexdigest()
        assert actual_hash == expected_hash, relative_path


def test_production_paths_can_be_supplied_without_checked_in_server_paths(
    tmp_path, monkeypatch
):
    overrides = {
        "IDEAL_DATABASE_URL": (
            "postgresql+psycopg://user:password@example.invalid/study"
        ),
        "IDEAL_STYLEGAN_REPO": str(tmp_path / "stylegan3"),
        "IDEAL_STYLEGAN_NETWORK": str(tmp_path / "stylegan3.pkl"),
        "IDEAL_FAIRFACE_RACE_WEIGHTS": str(tmp_path / "fairface_race"),
        "IDEAL_FAIRFACE_GENDER_WEIGHTS": str(tmp_path / "fairface_gender"),
        "IDEAL_FAIRFACE_AGE_WEIGHTS": str(tmp_path / "fairface_age"),
        "IDEAL_FACE_DETECTOR_PATH": str(tmp_path / "yunet.onnx"),
        "IDEAL_PRIOR_PATH": str(
            tmp_path / "priors" / "{gender}_20_29_{race}_d12.npz"
        ),
        "IDEAL_PERSONA_FEMALE_POOL": str(tmp_path / "female_pool"),
        "IDEAL_PERSONA_MALE_POOL": str(tmp_path / "male_pool"),
        "IDEAL_DATA_DIR": str(tmp_path / "data"),
        "IDEAL_CACHE_DIR": str(tmp_path / "cache"),
        "IDEAL_OUTPUT_DIR": str(tmp_path / "outputs"),
        "IDEAL_MODEL_DIR": str(tmp_path / "models"),
    }
    monkeypatch.delenv("IDEAL_DEMO", raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)

    config = load_config(
        PROJECT_ROOT / "configs" / "persona_study.yaml", demo_override=False
    )

    assert "/data1/" not in json.dumps(config.as_dict())
    assert config.database.url.startswith("postgresql+psycopg://")
    assert config.generator.network_path == str(tmp_path / "stylegan3.pkl")
