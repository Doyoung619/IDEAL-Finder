"""Vercel Python Function entrypoint for the dependency-light public gateway."""

from app.gateway import app


__all__ = ["app"]
