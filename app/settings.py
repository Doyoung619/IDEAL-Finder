from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def load_config(
    config_path: str | Path | None = None,
    demo_override: bool | None = None,
) -> ConfigNode:
    path = Path(
        config_path
        or os.getenv("IDEAL_CONFIG", PROJECT_ROOT / "configs" / "default.yaml")
    )
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    values = copy.deepcopy(raw)
    demo_enabled = (
        demo_override
        if demo_override is not None
        else os.getenv("IDEAL_DEMO", "0").lower() in {"1", "true", "yes"}
    )
    if demo_enabled:
        values["generator"]["mode"] = "demo"
        values["clip"]["enabled"] = False
        for key, value in values["demo"].items():
            if key in values["experiment"]:
                values["experiment"][key] = value

    for key in ("data_dir", "cache_dir", "output_dir", "model_dir"):
        values["paths"][key] = _resolve_project_path(values["paths"][key])
    for key in ("stylegan_repo", "network_path"):
        values["generator"][key] = _resolve_project_path(values["generator"][key])

    db_url = values["database"]["url"]
    if db_url.startswith("sqlite:///"):
        db_path = db_url.removeprefix("sqlite:///")
        values["database"]["url"] = f"sqlite:///{_resolve_project_path(db_path)}"

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

