#!/usr/bin/env bash
set -euo pipefail
export IDEAL_DEMO=1
export APP_ROLE=monolith
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
