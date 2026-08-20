#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

env_file="${IDEAL_ENV_FILE:-$repo_root/.env.gpu}"
if [[ -f "$env_file" ]]; then
  set -a
  source "$env_file"
  set +a
fi

fail() {
  echo "ERROR: $1" >&2
  exit 2
}

if [[ -n "${IDEAL_PYTHON:-}" ]]; then
  python_bin="$IDEAL_PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_bin="$repo_root/.venv/bin/python"
else
  python_bin="python"
fi

required_variables=(
  IDEAL_DATABASE_URL
  GPU_GATEWAY_SECRET
  SESSION_SECRET
  IDEAL_ADMIN_PASSWORD
  IDEAL_CONSENT_VERSION
  IDEAL_APP_VERSION
  IDEAL_STYLEGAN_REPO
  IDEAL_STYLEGAN_NETWORK
  IDEAL_PRIOR_PATH
  IDEAL_PERSONA_FEMALE_POOL
  IDEAL_PERSONA_MALE_POOL
)
for variable_name in "${required_variables[@]}"; do
  variable_value="${!variable_name:-}"
  [[ -n "$variable_value" ]] || fail "$variable_name is not configured."
  case "$variable_value" in
    replace-with-*|/path/to/*|postgresql+psycopg://USER:*|postgresql://USER:*)
      fail "$variable_name still contains the example placeholder."
      ;;
  esac
done

export APP_ROLE=gpu
export APP_ENV="${APP_ENV:-production}"
export IDEAL_CONFIG="${IDEAL_CONFIG:-configs/persona_study.yaml}"
export IDEAL_ARTIFACT_STORAGE="${IDEAL_ARTIFACT_STORAGE:-database}"

[[ "$APP_ENV" == "production" ]] || fail "APP_ENV must be production for the study worker."
[[ "$IDEAL_ARTIFACT_STORAGE" == "database" ]] || fail "IDEAL_ARTIFACT_STORAGE must be database."
[[ ${#GPU_GATEWAY_SECRET} -ge 32 ]] || fail "GPU_GATEWAY_SECRET must contain at least 32 characters."
[[ ${#SESSION_SECRET} -ge 32 ]] || fail "SESSION_SECRET must contain at least 32 characters."
[[ ${#IDEAL_ADMIN_PASSWORD} -ge 16 ]] || fail "IDEAL_ADMIN_PASSWORD must contain at least 16 characters."
case "$IDEAL_DATABASE_URL" in
  postgres://*|postgresql://*|postgresql+psycopg://*) ;;
  *) fail "IDEAL_DATABASE_URL must be a PostgreSQL connection string." ;;
esac

[[ -f "$IDEAL_CONFIG" ]] || fail "IDEAL_CONFIG does not exist: $IDEAL_CONFIG"
[[ -d "$IDEAL_STYLEGAN_REPO" ]] || fail "IDEAL_STYLEGAN_REPO is not a directory: $IDEAL_STYLEGAN_REPO"
[[ -f "$IDEAL_STYLEGAN_NETWORK" ]] || fail "IDEAL_STYLEGAN_NETWORK does not exist: $IDEAL_STYLEGAN_NETWORK"

for gender in female male; do
  prior_path="${IDEAL_PRIOR_PATH//\{gender\}/$gender}"
  prior_path="${prior_path//\{race\}/east_asian}"
  [[ -f "$prior_path" ]] || fail "IDEAL_PRIOR_PATH does not resolve to an existing $gender prior: $prior_path"
done

for pool_variable in IDEAL_PERSONA_FEMALE_POOL IDEAL_PERSONA_MALE_POOL; do
  pool_path="${!pool_variable}"
  [[ -d "$pool_path" ]] || fail "$pool_variable is not a directory: $pool_path"
  [[ -f "$pool_path/pool.npz" ]] || fail "$pool_variable is missing pool.npz: $pool_path"
  [[ -f "$pool_path/metadata.json" ]] || fail "$pool_variable is missing metadata.json: $pool_path"
done

if [[ "$python_bin" == */* ]]; then
  [[ -x "$python_bin" ]] || fail "IDEAL_PYTHON is not executable: $python_bin"
elif ! command -v "$python_bin" >/dev/null 2>&1; then
  fail "Python executable was not found: $python_bin"
fi

"$python_bin" -c "import fastapi, open_clip, sqlalchemy, torch, uvicorn" \
  || fail "The GPU Python environment is missing a required package. Run: python -m pip install -r requirements-gpu.txt"

host="${IDEAL_HOST:-0.0.0.0}"
port="${IDEAL_PORT:-8000}"

exec "$python_bin" -m uvicorn app.entrypoint:app \
  --host "$host" \
  --port "$port" \
  --workers 1
