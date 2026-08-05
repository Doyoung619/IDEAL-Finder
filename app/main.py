from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

from app.db import init_database
from app.routes import admin, participant
from app.services.runtime import ExperimentRuntime
from app.settings import PROJECT_ROOT, ConfigNode, load_config


def create_app(config: ConfigNode | None = None) -> FastAPI:
    settings = config or load_config()
    for directory in (
        settings.paths.data_dir,
        settings.paths.cache_dir,
        settings.paths.output_dir,
        settings.paths.model_dir,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
    init_database(settings.database.url)

    application = FastAPI(
        title=settings.app.name,
        docs_url="/api/docs" if settings.app.debug else None,
        redoc_url=None,
    )
    application.add_middleware(
        SessionMiddleware,
        secret_key=settings.app.secret_key,
        same_site="lax",
        https_only=False,
    )
    application.state.config = settings
    application.state.runtime = ExperimentRuntime(settings)
    application.state.templates = Jinja2Templates(
        directory=str(PROJECT_ROOT / "app" / "templates")
    )

    application.mount(
        "/static",
        StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")),
        name="static",
    )
    application.mount(
        "/generated",
        StaticFiles(directory=settings.paths.cache_dir),
        name="generated",
    )
    application.include_router(participant.router)
    application.include_router(admin.router)
    return application


app = create_app()
