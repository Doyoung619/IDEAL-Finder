from fastapi.testclient import TestClient

from app.db.base import reset_database_for_tests
from app.main import create_app
from app.settings import load_config


def test_persona_only_route_guards(tmp_path):
    config = load_config(demo_override=True)
    config.database._values["url"] = f"sqlite:///{tmp_path / 'guards.sqlite3'}"
    config.paths._values["data_dir"] = str(tmp_path / "data")
    config.paths._values["cache_dir"] = str(tmp_path / "cache")
    config.paths._values["output_dir"] = str(tmp_path / "outputs")
    config.paths._values["model_dir"] = str(tmp_path / "models")
    config.generator._values["output_resolution"] = 24
    reset_database_for_tests(config.database.url)
    app = create_app(config)

    with TestClient(app) as client:
        client.post("/consent", data={"consent": "yes"})
        client.post(
            "/basic-info",
            data={
                "age_band": "20s",
                "gender": "female",
                "preferred_target_gender": "male",
                "dating_experience": "none",
                "image_selection_importance": "3",
                "honest": "yes",
            },
        )
        candidates = client.get("/persona/candidates", follow_redirects=False)
        experiment = client.get(
            "/experiment1/instructions?block=0", follow_redirects=False
        )
        legacy_profile = client.get("/profile", follow_redirects=False)

        assert candidates.headers["location"] == "/persona"
        assert experiment.headers["location"] == "/persona"
        assert legacy_profile.headers["location"] == "/persona"
