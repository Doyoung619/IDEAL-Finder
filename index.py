import os


# Vercel must never import the Torch/StyleGAN application tree. Operators may
# still set APP_ROLE explicitly, but the public deployment defaults to gateway.
os.environ.setdefault("APP_ROLE", "gateway")

from app.entrypoint import app

__all__ = ["app"]
