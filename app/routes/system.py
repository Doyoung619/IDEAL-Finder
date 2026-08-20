from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db import get_db


router = APIRouter()


@router.get("/api/health")
def health(request: Request, db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            {"status": "unavailable", "database": "unavailable"},
            status_code=503,
        )
    payload = {
        "status": "ok",
        "database": "ok",
        "generation_mode": request.app.state.config.generator.mode,
    }
    if request.app.state.role == "gpu":
        payload["role"] = "gpu"
    return payload


@router.get("/internal/v1/health")
def internal_health(request: Request, db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        return JSONResponse(
            {
                "status": "unavailable",
                "role": request.app.state.role,
                "database": "unavailable",
            },
            status_code=503,
        )
    readiness = dict(request.app.state.readiness)
    readiness["database"] = "ok"
    return {
        "status": "ok",
        "role": request.app.state.role,
        **readiness,
    }
