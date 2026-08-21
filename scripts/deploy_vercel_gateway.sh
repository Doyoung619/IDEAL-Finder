#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() {
  echo "ERROR: $1" >&2
  exit 2
}

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    fail "Install shasum or sha256sum before deployment."
  fi
}

local_check() {
  command -v vercel >/dev/null 2>&1 \
    || fail "Vercel CLI is missing. Install it with: npm install -g vercel"
  command -v curl >/dev/null 2>&1 || fail "curl is required."
  [[ -f vercel.json ]] || fail "Run this command from the IDEAL-Finder repository."
  [[ -f index.py ]] || fail "index.py is missing."
  [[ -f requirements-vercel.txt ]] || fail "requirements-vercel.txt is missing."
  [[ -f configs/persona_study.yaml ]] || fail "The study config is missing."

  echo "Vercel: $(vercel --version 2>&1 | tail -n 1)"
  echo "Git commit: $(git rev-parse HEAD)"
  echo "Study config SHA-256: $(sha256_file configs/persona_study.yaml)"
}

set_vercel_value() {
  local name="$1"
  local value="$2"
  local sensitivity="$3"
  if [[ "$sensitivity" == "sensitive" ]]; then
    printf '%s\n' "$value" \
      | vercel env add "$name" production --force --yes --sensitive
  else
    printf '%s\n' "$value" \
      | vercel env add "$name" production --force --yes --no-sensitive
  fi
}

if [[ "${1:-}" == "--check" ]]; then
  local_check
  exit 0
fi
if [[ $# -ne 0 ]]; then
  fail "Usage: ./scripts/deploy_vercel_gateway.sh [--check]"
fi

local_check

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WARNING: The working tree has uncommitted changes."
  git status --short
  read -r -p "Continue with this exact working tree? Type CONTINUE: " dirty_confirm
  [[ "$dirty_confirm" == "CONTINUE" ]] || fail "Deployment cancelled."
fi

if ! vercel whoami >/dev/null 2>&1; then
  echo "Vercel login is required. Complete the browser login when prompted."
  vercel login
fi
vercel_user="$(vercel whoami | tail -n 1)"
echo "Vercel account: $vercel_user"

if [[ ! -f .vercel/project.json ]]; then
  echo "No linked Vercel project was found."
  echo "Choose your personal Hobby scope and create/link the intended project."
  vercel link
fi
[[ -f .vercel/project.json ]] || fail "Vercel project linking did not complete."

read -r -p "GPU_BACKEND_URL (HTTPS base URL only): " gpu_backend_url
gpu_backend_url="${gpu_backend_url%/}"
[[ "$gpu_backend_url" == https://* ]] \
  || fail "GPU_BACKEND_URL must start with https://"
[[ "$gpu_backend_url" != *"/internal/v1/health"* ]] \
  || fail "Enter the base URL without /internal/v1/health."

read -r -s -p "GPU_GATEWAY_SECRET (hidden, at least 32 characters): " gpu_gateway_secret
echo
[[ ${#gpu_gateway_secret} -ge 32 ]] \
  || fail "GPU_GATEWAY_SECRET must contain at least 32 characters."

echo "Checking the authenticated GPU endpoint before configuring Vercel..."
gpu_health="$(
  printf '%s\n' \
    'fail' \
    'show-error' \
    'silent' \
    "header = \"X-IDEAL-GATEWAY-SECRET: $gpu_gateway_secret\"" \
    "url = \"$gpu_backend_url/internal/v1/health\"" \
    | curl --config -
)" || fail "The GPU health check failed. Verify the URL, secret, HTTPS, and worker."
echo "GPU health: $gpu_health"

echo "Writing gateway-only Production variables to the linked Vercel project..."
set_vercel_value APP_ROLE "gateway" regular
set_vercel_value APP_ENV "production" regular
set_vercel_value GPU_BACKEND_URL "$gpu_backend_url" regular
set_vercel_value GPU_GATEWAY_SECRET "$gpu_gateway_secret" sensitive
set_vercel_value GPU_REQUEST_TIMEOUT_SECONDS "50" regular
set_vercel_value GPU_CONNECT_TIMEOUT_SECONDS "5" regular
unset gpu_gateway_secret

echo "Configured Production variable names:"
vercel env ls production

echo "Inspecting deployment inputs without uploading..."
vercel deploy --dry >/dev/null

echo "Release identity:"
echo "  commit=$(git rev-parse HEAD)"
echo "  config_sha256=$(sha256_file configs/persona_study.yaml)"
read -r -p "Create the Production deployment? Type DEPLOY: " deploy_confirm
[[ "$deploy_confirm" == "DEPLOY" ]] || {
  echo "Vercel variables are configured; Production deployment was not started."
  exit 0
}

deployment_output="$(vercel deploy --prod --yes)"
printf '%s\n' "$deployment_output"
deployment_url="$(printf '%s\n' "$deployment_output" | tail -n 1)"
[[ "$deployment_url" == https://* ]] \
  || fail "Vercel did not return a valid deployment URL."

echo "Waiting for the public gateway health check..."
for attempt in {1..12}; do
  if public_health="$(curl --fail --show-error --silent "$deployment_url/api/health")"; then
    echo "Public health: $public_health"
    echo "Production URL: $deployment_url"
    echo "Next: complete one non-study participant, verify PostgreSQL, and export CSV."
    exit 0
  fi
  if [[ "$attempt" -lt 12 ]]; then
    sleep 5
  fi
done

fail "Production deployed, but /api/health did not become healthy: $deployment_url/api/health"
