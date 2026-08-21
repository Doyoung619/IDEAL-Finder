from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.base import get_session, reset_database_for_tests
from app.gateway import FORWARDED_SECRET_HEADER, create_gateway_app
from app.main import create_app
from app.models import ExperimentRound, ExperimentSession, Participant, Selection
from app.services.export_service import build_csv_export_zip
from app.services.persona_service import PERSONA_CATEGORIES
from app.settings import load_config


SECRET = "test-gateway-secret-that-is-not-used-in-production"


def _hidden_value(html: str, name: str) -> str:
    match = re.search(
        rf'<input[^>]+name="{re.escape(name)}"[^>]+value="([^"]+)"', html
    )
    assert match is not None
    return match.group(1)


def _first_image_id(html: str) -> str:
    match = re.search(r'data-image-id="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def _image_ids(html: str) -> set[str]:
    return set(re.findall(r'data-image-id="([^"]+)"', html))


def _mock_stack(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GPU_GATEWAY_SECRET", SECRET)
    config = load_config(demo_override=True)
    config.database._values["url"] = f"sqlite:///{tmp_path / 'gateway.sqlite3'}"
    for name in ("data_dir", "cache_dir", "output_dir", "model_dir"):
        config.paths._values[name] = str(tmp_path / name)
    config.app._values["secure_cookies"] = False
    config.generator._values["output_resolution"] = 16
    config.search._values["pca_dimensions"] = 4
    config.search._values["pca_fit_samples"] = 8
    config.experiment._values["m_list"] = [2]
    config.experiment._values["algorithm_order"] = ["entropy"]
    config.experiment._values["rounds_per_m"] = 2
    config.experiment._values["candidate_pool_size"] = 16
    config.experiment._values["candidate_pool_multiplier"] = 2
    config.persona._values["mixture_particle_count"] = 512
    config.persona._values["max_candidate_pages"] = 2
    config.query._values["posterior_mc_samples"] = 32
    config.query._values["rc_posterior_samples"] = 512
    config.query._values["num_restarts"] = 1
    config.query._values["optimization_steps"] = 5
    reset_database_for_tests(config.database.url)
    gpu_app = create_app(config, role="gpu")
    gateway_app = create_gateway_app(
        backend_url="http://gpu.internal",
        gateway_secret=SECRET,
        transport=httpx.ASGITransport(app=gpu_app),
    )
    return config, gpu_app, gateway_app


def _complete_persona(client: TestClient) -> None:
    client.post("/consent", data={"consent": "yes"})
    client.post(
        "/basic-info",
        data={
            "age_band": "20s",
            "gender": "female",
            "preferred_target_gender": "male",
            "preferred_age_appearance": "twenties_boost",
            "honest": "yes",
        },
    )
    client.post(
        "/persona",
        data={
            f"persona::{category.key}": "no_preference"
            for category in PERSONA_CATEGORIES
        },
    )
    first_page = client.get("/persona/candidates")
    first_ids = _image_ids(first_page.text)
    assert first_ids
    client.post(
        "/persona/candidates",
        data={
            "batch_id": _hidden_value(first_page.text, "batch_id"),
            "action": "show_more",
            "reaction_time_sec": "0.4",
        },
    )
    page = client.get("/persona/candidates")
    assert first_ids.isdisjoint(_image_ids(page.text))
    assert 'value="show_more"' not in page.text
    client.post(
        "/persona/candidates",
        data={
            "batch_id": _hidden_value(page.text, "batch_id"),
            "selected_image_id": _first_image_id(page.text),
            "action": "selected",
            "reaction_time_sec": "0.5",
        },
    )
    client.post(
        "/persona/confirm",
        data={"initial_rating": "7", "selection_confidence": "5"},
    )


def _submit_round(client: TestClient, round_page) -> None:
    client.post(
        "/experiment1/round",
        data={
            "block_index": "0",
            "round_id": _hidden_value(round_page.text, "round_id"),
            "selected_image_id": _first_image_id(round_page.text),
            "preference_rating": "8",
            "difficulty": "3",
            "reaction_time_sec": "1.0",
        },
    )


def test_gpu_authentication_and_gateway_health(tmp_path, monkeypatch):
    _config, gpu_app, gateway_app = _mock_stack(tmp_path, monkeypatch)
    with TestClient(gpu_app) as direct:
        assert direct.get("/internal/v1/health").status_code == 401
        assert direct.get(
            "/internal/v1/health",
            headers={FORWARDED_SECRET_HEADER: "wrong"},
        ).status_code == 403
        accepted = direct.get(
            "/internal/v1/health",
            headers={FORWARDED_SECRET_HEADER: SECRET},
        )
        assert accepted.status_code == 200
        assert accepted.json()["role"] == "gpu"

    with TestClient(gateway_app) as gateway:
        health = gateway.get("/api/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "role": "gateway",
            "gpu": "reachable",
        }


def test_gateway_mock_end_to_end_resume_completion_and_csv(tmp_path, monkeypatch):
    _config, _gpu_app, gateway_app = _mock_stack(tmp_path, monkeypatch)
    with TestClient(gateway_app) as client:
        landing = client.get("/")
        assert landing.status_code == 200
        assert "GPU_GATEWAY_SECRET" not in landing.text
        javascript = client.get("/static/app.js")
        assert javascript.status_code == 200
        assert SECRET not in javascript.text
        assert "GPU_GATEWAY_SECRET" not in javascript.text
        _complete_persona(client)

        client.post("/experiment1/start", data={"block_index": "0"})
        first = client.get("/experiment1/round?block=0")
        assert first.status_code == 200
        image_url = re.search(r'<img src="([^"]+)"', first.text).group(1)
        assert "gpu.internal" not in image_url
        assert client.get(image_url).headers["content-type"] == "image/png"
        _submit_round(client, first)

        second = client.get("/experiment1/round?block=0")
        assert _hidden_value(second.text, "round_id") == "2"
        refreshed = client.get("/experiment1/round?block=0")
        assert _first_image_id(refreshed.text) == _first_image_id(second.text)
        _submit_round(client, second)

        final_page = client.get("/experiment1/round?block=0")
        assert final_page.url.path == "/final-evaluation"
        preferred = _first_image_id(final_page.text)
        client.post(
            "/final-evaluation",
            data={
                "preferred_image_id": preferred,
                "initial_rating": "7",
                "final_rating": "8",
                "perceived_improvement": "6",
                "final_match": "8",
                "reaction_time_sec": "1.0",
            },
        )
        completed = client.post(
            "/survey",
            data={"q1": "6", "q2": "6", "q3": "6", "q4": "6"},
        )
        assert completed.url.path == "/complete"

    with get_session() as db:
        participant = db.scalar(select(Participant))
        experiment_session = db.scalar(select(ExperimentSession))
        assert participant.status == "completed"
        assert experiment_session.status == "completed"
        assert db.scalar(select(func.count(Selection.id))) == 2
        assert db.scalar(select(func.count(ExperimentRound.round_id))) == 2
        archive_bytes = build_csv_export_zip(db)

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        required = {
            "sessions.csv",
            "blocks.csv",
            "rounds.csv",
            "events.csv",
            "rounds_flat.csv",
            "analysis_rounds.csv",
        }
        assert required.issubset(archive.namelist())
        assert participant.participant_id in archive.read("sessions.csv").decode()
        assert participant.participant_id in archive.read("rounds_flat.csv").decode()


def test_vercel_entrypoint_imports_no_heavy_modules():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os,sys; os.environ['APP_ROLE']='gateway'; "
                "os.environ['GPU_BACKEND_URL']='http://localhost:9000'; "
                "os.environ['GPU_GATEWAY_SECRET']='test'; import index; "
                "blocked={'torch','sqlalchemy','numpy','PIL','open_clip'}; "
                "loaded={name.split('.')[0] for name in sys.modules}; "
                "assert not blocked & loaded, blocked & loaded"
            ),
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_vercel_config_uses_api_fastapi_function():
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads((project_root / "vercel.json").read_text(encoding="utf-8"))

    assert "installCommand" not in config
    assert config["functions"] == {"api/index.py": {"maxDuration": 60}}
    assert config["rewrites"] == [{"source": "/(.*)", "destination": "/api"}]

    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"fastapi>=0.115,<1"' in pyproject
    assert '"httpx>=0.27,<1"' in pyproject


def test_vercel_api_entrypoint_imports_no_heavy_modules():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os,sys; os.environ['APP_ENV']='production'; "
                "os.environ['GPU_BACKEND_URL']='https://gpu.example.invalid'; "
                "os.environ['GPU_GATEWAY_SECRET']='x'*32; import api.index; "
                "blocked={'torch','sqlalchemy','numpy','PIL','open_clip'}; "
                "loaded={name.split('.')[0] for name in sys.modules}; "
                "assert not blocked & loaded, blocked & loaded"
            ),
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_gateway_returns_retryable_errors_for_gpu_failures():
    def unavailable(request: httpx.Request):
        raise httpx.ConnectError("offline", request=request)

    disconnected = create_gateway_app(
        backend_url="http://gpu.internal",
        gateway_secret=SECRET,
        transport=httpx.MockTransport(unavailable),
    )
    with TestClient(disconnected) as client:
        response = client.get("/")
        assert response.status_code == 503
        assert response.headers["retry-after"] == "3"
        assert "다시 시도" in response.text

    def timed_out(request: httpx.Request):
        raise httpx.ReadTimeout("slow", request=request)

    slow = create_gateway_app(
        backend_url="http://gpu.internal",
        gateway_secret=SECRET,
        transport=httpx.MockTransport(timed_out),
    )
    with TestClient(slow) as client:
        response = client.get("/")
        assert response.status_code == 504
        assert response.headers["retry-after"] == "3"


def test_production_session_cookie_flags(tmp_path, monkeypatch):
    _config, gpu_app, _gateway_app = _mock_stack(tmp_path, monkeypatch)
    gpu_app.state.config.app._values["secure_cookies"] = True
    # Middleware captured the setting during creation; build once more with Secure enabled.
    secure_app = create_app(gpu_app.state.config, role="gpu")
    with TestClient(secure_app) as client:
        response = client.post(
            "/consent",
            data={"consent": "yes"},
            headers={FORWARDED_SECRET_HEADER: SECRET},
            follow_redirects=False,
        )
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "secure" in cookie
    assert "samesite=lax" in cookie
