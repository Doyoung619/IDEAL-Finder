from __future__ import annotations

import secrets

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


GATEWAY_SECRET_HEADER = b"x-ideal-gateway-secret"


class GatewayAuthenticationMiddleware:
    """Require the Vercel-to-GPU shared secret on every GPU HTTP route."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        secret: str,
        production: bool = False,
    ) -> None:
        if not secret:
            raise RuntimeError("GPU_GATEWAY_SECRET is required when APP_ROLE=gpu")
        if production and len(secret) < 32:
            raise RuntimeError(
                "GPU_GATEWAY_SECRET must contain at least 32 characters in production"
            )
        self.app = app
        self.secret = secret

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        provided = next(
            (
                value.decode("utf-8", errors="ignore")
                for name, value in scope.get("headers", [])
                if name.lower() == GATEWAY_SECRET_HEADER
            ),
            None,
        )
        if provided is None:
            response = JSONResponse(
                {"detail": "GPU gateway authentication is required"},
                status_code=401,
            )
            await response(scope, receive, send)
            return
        if not secrets.compare_digest(provided, self.secret):
            response = JSONResponse(
                {"detail": "GPU gateway authentication failed"},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
