#!/usr/bin/env bash
set -euo pipefail
unset IDEAL_DEMO || true
export APP_ROLE=monolith
python scripts/invalidate_legacy_sessions.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
