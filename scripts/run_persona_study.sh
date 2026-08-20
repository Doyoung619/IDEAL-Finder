#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${IDEAL_PYTHON:-}" ]]; then
  python_bin="$IDEAL_PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin="python"
fi

export IDEAL_CONFIG="${IDEAL_CONFIG:-configs/persona_study.yaml}"
export APP_ROLE=monolith
host="${IDEAL_HOST:-0.0.0.0}"
port="${IDEAL_PORT:-8000}"

"$python_bin" scripts/validate_persona_pool.py \
  "${IDEAL_PERSONA_FEMALE_POOL:-artifacts/persona_pools/female_20_29_east_asian_d12}" \
  --minimum-size 1000
"$python_bin" scripts/validate_persona_pool.py \
  "${IDEAL_PERSONA_MALE_POOL:-artifacts/persona_pools/male_20_29_east_asian_d12}" \
  --minimum-size 1000
"$python_bin" scripts/invalidate_legacy_sessions.py

exec "$python_bin" -m uvicorn app.main:app --host "$host" --port "$port"
