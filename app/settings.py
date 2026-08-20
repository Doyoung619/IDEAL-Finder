from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
DEFAULT_SECRET_KEY = "change-this-before-data-collection"


class ConfigNode(Mapping[str, Any]):
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = {
            key: ConfigNode(value) if isinstance(value, dict) else value
            for key, value in values.items()
        }

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self._values.items():
            result[key] = value.as_dict() if isinstance(value, ConfigNode) else value
        return result


def _resolve_project_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise RuntimeError(
                f"Required environment variable '{name}' is not set"
            )
        return os.environ[name]

    return ENVIRONMENT_PATTERN.sub(replace, value)


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _running_on_vercel() -> bool:
    return _truthy(os.getenv("VERCEL")) or _truthy(os.getenv("IDEAL_SERVERLESS"))


def _production_mode() -> bool:
    return os.getenv("APP_ENV", "development").strip().lower() == "production"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: str | Path | None = None,
    demo_override: bool | None = None,
) -> ConfigNode:
    serverless = _running_on_vercel()
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    path = Path(
        config_path
        or os.getenv("IDEAL_CONFIG", default_path)
    )
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if path.resolve() != default_path.resolve():
        with default_path.open("r", encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle)
        raw = _deep_merge(defaults, raw)
    values = _expand_environment(copy.deepcopy(raw))
    demo_enabled = (
        demo_override
        if demo_override is not None
        else (
            _truthy(os.getenv("IDEAL_DEMO"))
            or (serverless and "IDEAL_DEMO" not in os.environ)
        )
    )
    if demo_enabled:
        values["generator"]["mode"] = "demo"
        values["clip"]["enabled"] = False
        for key, value in values["demo"].items():
            if key in values["experiment"]:
                values["experiment"][key] = value
            elif key == "query":
                values["query"].update(value)
            elif key == "mixture_particle_count":
                values["persona"][key] = value
        values["persona"]["require_real_clip"] = False
        values["persona_pool"]["target_size"] = int(
            values["demo"].get("persona_pool_target_size", 32)
        )
        values["persona_pool"]["minimum_usable_size"] = int(
            values["demo"].get("persona_pool_minimum_usable_size", 8)
        )

    session_secret = os.getenv("SESSION_SECRET") or os.getenv("IDEAL_SECRET_KEY")
    if session_secret:
        values["app"]["secret_key"] = session_secret
    if os.getenv("IDEAL_CONSENT_VERSION"):
        values["app"]["consent_version"] = os.environ[
            "IDEAL_CONSENT_VERSION"
        ]
    if os.getenv("IDEAL_SECURE_COOKIES"):
        values["app"]["secure_cookies"] = _truthy(
            os.getenv("IDEAL_SECURE_COOKIES")
        )
    if os.getenv("IDEAL_DEBUG"):
        values["app"]["debug"] = _truthy(os.getenv("IDEAL_DEBUG"))
    if os.getenv("IDEAL_DATABASE_URL"):
        values["database"]["url"] = os.environ["IDEAL_DATABASE_URL"]
    if os.getenv("IDEAL_GENERATOR_MODE"):
        values["generator"]["mode"] = os.environ["IDEAL_GENERATOR_MODE"]
    for key, env_name in (
        ("stylegan_repo", "IDEAL_STYLEGAN_REPO"),
        ("network_path", "IDEAL_STYLEGAN_NETWORK"),
    ):
        if os.getenv(env_name):
            values["generator"][key] = os.environ[env_name]
    for key, env_name in (
        ("race_weights", "IDEAL_FAIRFACE_RACE_WEIGHTS"),
        ("gender_weights", "IDEAL_FAIRFACE_GENDER_WEIGHTS"),
        ("age_weights", "IDEAL_FAIRFACE_AGE_WEIGHTS"),
    ):
        if os.getenv(env_name):
            values["demographic"][key] = os.environ[env_name]
    if os.getenv("IDEAL_PRIOR_PATH"):
        values["conditional_prior"]["artifact_path"] = os.environ[
            "IDEAL_PRIOR_PATH"
        ]
    for key, env_name in (
        ("female_pool", "IDEAL_PERSONA_FEMALE_POOL"),
        ("male_pool", "IDEAL_PERSONA_MALE_POOL"),
    ):
        if os.getenv(env_name):
            values["persona"][key] = os.environ[env_name]
    if os.getenv("IDEAL_FACE_DETECTOR_PATH"):
        values["persona_pool"]["face_detector_path"] = os.environ[
            "IDEAL_FACE_DETECTOR_PATH"
        ]
    if os.getenv("IDEAL_ARTIFACT_STORAGE"):
        values.setdefault("storage", {})["persist_artifacts_in_db"] = (
            os.environ["IDEAL_ARTIFACT_STORAGE"].lower() == "database"
        )
    elif serverless:
        values.setdefault("storage", {})["persist_artifacts_in_db"] = True

    for key, env_name in (
        ("data_dir", "IDEAL_DATA_DIR"),
        ("cache_dir", "IDEAL_CACHE_DIR"),
        ("output_dir", "IDEAL_OUTPUT_DIR"),
        ("model_dir", "IDEAL_MODEL_DIR"),
    ):
        if os.getenv(env_name):
            values["paths"][key] = os.environ[env_name]
        elif serverless:
            values["paths"][key] = f"/tmp/ideal-finder/{key}"

    for key in ("data_dir", "cache_dir", "output_dir", "model_dir"):
        values["paths"][key] = _resolve_project_path(values["paths"][key])
    for key in ("stylegan_repo", "network_path"):
        values["generator"][key] = _resolve_project_path(values["generator"][key])
    if "demographic" in values:
        values["demographic"]["weights"] = _resolve_project_path(
            values["demographic"]["weights"]
        )
        for key in ("race_weights", "gender_weights", "age_weights"):
            if key in values["demographic"]:
                values["demographic"][key] = _resolve_project_path(
                    values["demographic"][key]
                )
    if "conditional_prior" in values:
        values["conditional_prior"]["artifact_path"] = _resolve_project_path(
            values["conditional_prior"]["artifact_path"]
        )
    if "persona" in values:
        for key in ("pool_root", "female_pool", "male_pool"):
            values["persona"][key] = _resolve_project_path(values["persona"][key])
    if "persona_pool" in values and values["persona_pool"].get(
        "face_detector_path"
    ):
        values["persona_pool"]["face_detector_path"] = _resolve_project_path(
            values["persona_pool"]["face_detector_path"]
        )

    db_url = values["database"]["url"]
    if db_url.startswith("postgres://"):
        db_url = "postgresql+psycopg://" + db_url.removeprefix("postgres://")
    elif db_url.startswith("postgresql://"):
        db_url = "postgresql+psycopg://" + db_url.removeprefix("postgresql://")
    if db_url.startswith("sqlite:///"):
        db_path = db_url.removeprefix("sqlite:///")
        db_url = f"sqlite:///{_resolve_project_path(db_path)}"
    values["database"]["url"] = db_url

    production = _production_mode()
    if serverless or production:
        if values["app"]["secret_key"] == DEFAULT_SECRET_KEY:
            raise RuntimeError(
                "SESSION_SECRET (or legacy IDEAL_SECRET_KEY) must be set in production."
            )
        values["app"]["secure_cookies"] = True
        if (
            os.getenv("APP_ROLE", "monolith").strip().lower() != "gateway"
            and values["database"]["url"].startswith("sqlite")
        ):
            raise RuntimeError(
                "IDEAL_DATABASE_URL must point to an external database for "
                "production GPU/monolith deployments."
            )

    for directory in values["paths"].values():
        Path(directory).mkdir(parents=True, exist_ok=True)

    return ConfigNode(values)


def save_config_snapshot(config: ConfigNode, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            json.loads(json.dumps(config.as_dict())),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
