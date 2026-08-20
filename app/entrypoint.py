from __future__ import annotations

import os


role = os.getenv("APP_ROLE", "monolith").strip().lower()
if role == "gateway":
    from app.gateway import app
elif role in {"gpu", "monolith"}:
    from app.main import app
else:
    raise RuntimeError("APP_ROLE must be one of: gateway, gpu, monolith")


__all__ = ["app"]
