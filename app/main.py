from __future__ import annotations

import os
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

from app.db import init_database
from app.routes import admin, participant, system
from app.services.runtime import ExperimentRuntime
from app.services.startup_validation import validate_gpu_startup
from app.security import GatewayAuthenticationMiddleware
from app.settings import PROJECT_ROOT, ConfigNode, load_config


logger = logging.getLogger("ideal_finder.requests")


def create_app(
    config: ConfigNode | None = None,
    *,
    role: str | None = None,
) -> FastAPI:
    selected_role = (role or os.getenv("APP_ROLE", "monolith")).strip().lower()
    if selected_role not in {"gpu", "monolith"}:
        raise ValueError("app.main can only create the gpu or monolith application")
    settings = config or load_config()
    for directory in (
        settings.paths.data_dir,
        settings.paths.cache_dir,
        settings.paths.output_dir,
        settings.paths.model_dir,
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)
    init_database(settings.database.url)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if selected_role == "gpu":
            application.state.readiness = validate_gpu_startup(
                settings,
                application.state.runtime,
            )
        yield

    application = FastAPI(
        title=settings.app.name,
        docs_url="/api/docs" if settings.app.debug else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(
        SessionMiddleware,
        secret_key=settings.app.secret_key,
        same_site="lax",
        https_only=bool(settings.app.secure_cookies),
        max_age=60 * 60 * 24 * 14,
    )
    if selected_role == "gpu":
        application.add_middleware(
            GatewayAuthenticationMiddleware,
            secret=os.getenv("GPU_GATEWAY_SECRET", ""),
            production=(
                os.getenv("APP_ENV", "development").strip().lower()
                == "production"
            ),
        )
    application.state.config = settings
    application.state.role = selected_role
    application.state.runtime = ExperimentRuntime(settings)
    application.state.readiness = {
        "database": "unchecked",
        "cuda": False,
        "generator_loaded": False,
        "openclip_loaded": False,
        "artifacts": "unchecked",
    }
    application.state.templates = Jinja2Templates(
        directory=str(PROJECT_ROOT / "app" / "templates")
    )

    @application.middleware("http")
    async def request_latency(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["Server-Timing"] = f"ideal;dur={elapsed_ms:.1f}"
        logger.info(
            "request_latency path=%s method=%s status=%s "
            "total_request_latency_ms=%.1f",
            request.url.path,
            request.method,
            response.status_code,
            elapsed_ms,
        )
        return response

    @application.exception_handler(SQLAlchemyError)
    async def database_unavailable(_request, _error):
        return HTMLResponse(
            "<h1>임시로 저장 서버에 연결할 수 없습니다.</h1>"
            "<p>이 화면에서 다음 단계로 넘어가지 않았습니다. "
            "잠시 후 브라우저의 뒤로 가기 후 다시 제출해 주세요.</p>",
            status_code=503,
        )

    application.mount(
        "/static",
        StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")),
        name="static",
    )
    application.include_router(participant.router)
    application.include_router(admin.router)
    application.include_router(system.router)

    return application


app = create_app()
