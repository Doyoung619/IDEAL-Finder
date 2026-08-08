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

    if os.getenv("IDEAL_SECRET_KEY"):
        values["app"]["secret_key"] = os.environ["IDEAL_SECRET_KEY"]
    if os.getenv("IDEAL_DEBUG"):
        values["app"]["debug"] = _truthy(os.getenv("IDEAL_DEBUG"))
    if os.getenv("IDEAL_DATABASE_URL"):
        values["database"]["url"] = os.environ["IDEAL_DATABASE_URL"]
    if os.getenv("IDEAL_GENERATOR_MODE"):
        values["generator"]["mode"] = os.environ["IDEAL_GENERATOR_MODE"]
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
    if "conditional_prior" in values:
        values["conditional_prior"]["artifact_path"] = _resolve_project_path(
            values["conditional_prior"]["artifact_path"]
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

    if serverless:
        if values["app"]["secret_key"] == DEFAULT_SECRET_KEY:
            raise RuntimeError(
                "IDEAL_SECRET_KEY must be set for serverless/Vercel deployments."
            )
        if values["database"]["url"].startswith("sqlite"):
            raise RuntimeError(
                "IDEAL_DATABASE_URL must point to an external database for "
                "serverless/Vercel deployments."
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
