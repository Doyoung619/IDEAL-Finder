from __future__ import annotations

import html
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


FORWARDED_SECRET_HEADER = "X-IDEAL-GATEWAY-SECRET"
HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
PROXY_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD")


def create_gateway_app(
    *,
    backend_url: str | None = None,
    gateway_secret: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create the dependency-light Vercel gateway without importing app.main."""
    configured_url = (backend_url or os.getenv("GPU_BACKEND_URL", "")).rstrip("/")
    configured_secret = gateway_secret or os.getenv("GPU_GATEWAY_SECRET", "")
    request_timeout = float(os.getenv("GPU_REQUEST_TIMEOUT_SECONDS", "50"))
    connect_timeout = float(os.getenv("GPU_CONNECT_TIMEOUT_SECONDS", "5"))

    application = FastAPI(
        title="IDEAL-Finder Gateway",
        docs_url=None,
        redoc_url=None,
    )
    application.state.role = "gateway"
    application.state.backend_url = configured_url
    application.state.gateway_secret = configured_secret
    application.state.transport = transport
    application.state.timeout = httpx.Timeout(
        request_timeout,
        connect=min(connect_timeout, request_timeout),
    )
    application.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).resolve().parent / "static")),
        name="static",
    )

    @application.get("/api/health")
    async def gateway_health():
        if not _configuration_ready(application):
            return JSONResponse(
                {
                    "status": "unavailable",
                    "role": "gateway",
                    "gpu": "not_configured",
                },
                status_code=503,
            )
        try:
            upstream = await _request_backend(
                application,
                method="GET",
                path="/internal/v1/health",
                query=b"",
                headers={},
                content=b"",
            )
        except (httpx.TimeoutException, httpx.TransportError):
            return JSONResponse(
                {
                    "status": "degraded",
                    "role": "gateway",
                    "gpu": "unreachable",
                },
                status_code=503,
            )
        if upstream.status_code != 200:
            return JSONResponse(
                {
                    "status": "degraded",
                    "role": "gateway",
                    "gpu": "unavailable",
                },
                status_code=503,
            )
        return {
            "status": "ok",
            "role": "gateway",
            "gpu": "reachable",
        }

    @application.api_route("/{path:path}", methods=PROXY_METHODS)
    async def proxy_to_gpu(path: str, request: Request):
        if not _configuration_ready(application):
            return _retry_page(
                "GPU backend가 아직 설정되지 않았습니다.",
                status_code=503,
            )
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower()
            not in {"content-length", "host", FORWARDED_SECRET_HEADER.lower()}
        }
        headers["x-forwarded-proto"] = request.url.scheme
        if request.headers.get("host"):
            headers["x-forwarded-host"] = request.headers["host"]
        try:
            upstream = await _request_backend(
                application,
                method=request.method,
                path=request.url.path,
                query=request.url.query.encode("ascii"),
                headers=headers,
                content=await request.body(),
            )
        except httpx.TimeoutException:
            return _retry_page(
                "얼굴 생성 시간이 길어지고 있습니다. 저장된 진행 상태는 유지됩니다.",
                status_code=504,
            )
        except httpx.TransportError:
            return _retry_page(
                "GPU 서버에 일시적으로 연결할 수 없습니다. 저장된 진행 상태는 유지됩니다.",
                status_code=503,
            )
        return _gateway_response(upstream, application.state.backend_url)

    return application


def _configuration_ready(application: FastAPI) -> bool:
    configured = bool(
        application.state.backend_url
        and application.state.gateway_secret
        and application.state.backend_url.startswith(("http://", "https://"))
    )
    if not configured:
        return False
    if os.getenv("APP_ENV", "development").strip().lower() == "production":
        return bool(
            application.state.backend_url.startswith("https://")
            and len(application.state.gateway_secret) >= 32
        )
    return True


async def _request_backend(
    application: FastAPI,
    *,
    method: str,
    path: str,
    query: bytes,
    headers: dict[str, str],
    content: bytes,
) -> httpx.Response:
    target = f"{application.state.backend_url}{path}"
    if query:
        target = f"{target}?{query.decode('ascii')}"
    forwarded_headers = dict(headers)
    forwarded_headers[FORWARDED_SECRET_HEADER] = application.state.gateway_secret
    async with httpx.AsyncClient(
        transport=application.state.transport,
        timeout=application.state.timeout,
        follow_redirects=False,
    ) as client:
        return await client.request(
            method,
            target,
            headers=forwarded_headers,
            content=content,
        )


def _gateway_response(upstream: httpx.Response, backend_url: str) -> Response:
    response = Response(content=upstream.content, status_code=upstream.status_code)
    response.raw_headers = []
    for name, value in upstream.headers.multi_items():
        lower_name = name.lower()
        if lower_name in HOP_BY_HOP_RESPONSE_HEADERS:
            continue
        if lower_name == "location":
            value = _safe_location(value, backend_url)
        response.headers.append(name, value)
    if "content-type" not in response.headers:
        response.headers["content-type"] = "application/octet-stream"
    response.headers["content-length"] = str(len(upstream.content))
    return response


def _safe_location(value: str, backend_url: str) -> str:
    if not value.startswith(backend_url):
        return value
    parsed = urlsplit(value)
    location = parsed.path or "/"
    if parsed.query:
        location = f"{location}?{parsed.query}"
    return location


def _retry_page(message: str, *, status_code: int) -> HTMLResponse:
    safe_message = html.escape(message)
    return HTMLResponse(
        "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>IDEAL-Finder 일시적 오류</title></head><body>"
        "<main><h1>잠시 후 다시 시도해 주세요.</h1>"
        f"<p>{safe_message}</p>"
        "<button type=\"button\" onclick=\"location.reload()\">다시 시도</button>"
        "</main></body></html>",
        status_code=status_code,
        headers={"Retry-After": "3", "Cache-Control": "no-store"},
    )


app = create_gateway_app()
