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

gateway_secret="${GPU_GATEWAY_SECRET:-local-development-gateway-secret}"
mock_port="${IDEAL_MOCK_GPU_PORT:-9000}"
gateway_port="${IDEAL_GATEWAY_PORT:-3000}"
mock_database="${IDEAL_MOCK_DATABASE_PATH:-$repo_root/data/mock_gpu.sqlite3}"

cleanup() {
  if [[ -n "${mock_pid:-}" ]]; then
    kill "$mock_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

GPU_GATEWAY_SECRET="$gateway_secret" \
  "$python_bin" scripts/run_mock_gpu_backend.py \
  --host 127.0.0.1 --port "$mock_port" --database "$mock_database" &
mock_pid=$!

mock_ready=0
for _attempt in $(seq 1 60); do
  if curl --fail --silent \
    -H "X-IDEAL-GATEWAY-SECRET: $gateway_secret" \
    "http://127.0.0.1:$mock_port/internal/v1/health" >/dev/null; then
    mock_ready=1
    break
  fi
  if ! kill -0 "$mock_pid" 2>/dev/null; then
    echo "Mock GPU backend stopped during startup." >&2
    exit 1
  fi
  sleep 0.25
done
if [[ "$mock_ready" != "1" ]]; then
  echo "Mock GPU backend did not become healthy in time." >&2
  exit 1
fi

echo "IDEAL-Finder gateway: http://127.0.0.1:$gateway_port"
APP_ROLE=gateway \
APP_ENV=development \
GPU_BACKEND_URL="http://127.0.0.1:$mock_port" \
GPU_GATEWAY_SECRET="$gateway_secret" \
  "$python_bin" -m uvicorn app.entrypoint:app \
  --host 127.0.0.1 --port "$gateway_port" --workers 1
