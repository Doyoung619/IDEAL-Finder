# GPU server handoff

이 문서는 연구실 GPU 서버 담당자가 받는 독립 실행 안내서입니다. GPU worker는
StyleGAN3/OpenCLIP, participant별 posterior와 PostgreSQL 저장을 담당하고, Vercel은
인증된 public gateway로만 동작합니다.

## Quick Start (5 steps)

1. 연구자가 전달한 release commit을 checkout하고 Python 3.10 virtualenv를 만듭니다.
2. `python -m pip install -r requirements-gpu.txt`를 실행합니다.
3. `cp .env.gpu.example .env.gpu` 후 전달받은 DB URL·secret·release/version과 서버의 실제 artifact 경로를 입력합니다.
4. `./scripts/run_gpu_backend.sh`를 실행하고 아래 인증 health check가 성공하는지 확인합니다.
5. 승인된 HTTPS endpoint를 붙인 뒤 연구자에게 `GPU_BACKEND_URL=https://...`만 회신합니다.

DB 생성·migration, secret 생성, Vercel 설정은 연구자가 수행합니다. GPU 담당자는
전달받은 값을 `.env.gpu`에 입력하며 실제 값이나 연구 데이터를 Git에 commit하지
않습니다.

## Runtime and hardware

- 이 release의 기준 Python은 3.10입니다.
- `requirements.txt`의 PyTorch 범위와 연구실 NVIDIA driver/CUDA 정책에 맞는 wheel을 사용합니다.
- 한 CUDA GPU에서 시작하며 `uvicorn --workers 1`을 유지합니다.
- VRAM과 disk 필요량은 checkpoint, pool, CUDA build/cache 및 실제 batch에 따라 달라집니다. 수집 전 대상 서버에서 startup과 한 번의 비연구 session으로 측정하고 여유 용량을 확보하세요.
- Vercel에서 접근할 수 있는 안정적인 HTTPS endpoint가 필요합니다.

Docker image는 제공하지 않습니다. StyleGAN3 CUDA extension과 PyTorch wheel은 연구실
driver/toolkit에 맞아야 하므로 기존 서버 설치 정책을 따릅니다.

## Required artifacts

- StyleGAN3 source directory와 FFHQ-U checkpoint
- female/male `20_29_east_asian_d12.npz` conditional priors
- female/male Persona pool directory(`pool.npz`, `metadata.json`, image files)
- 설정된 OpenCLIP model/checkpoint cache

FairFace triplet weights와 YuNet은 offline prior/pool 재구축·재검증 때 필요합니다.
온라인 worker는 검증이 끝난 prior/pool을 사용하므로 이 경로들은 해당 작업을 할 때만
필요합니다.

## Install and configure

```bash
git clone <repository-url> IDEAL-Finder
cd IDEAL-Finder
git checkout <release-commit>
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
cp .env.gpu.example .env.gpu
```

`.env.gpu`에 연구자가 안전한 채널로 전달한 다음 값을 입력합니다.

- `IDEAL_DATABASE_URL`, `GPU_GATEWAY_SECRET`, `SESSION_SECRET`, `IDEAL_ADMIN_PASSWORD`
- `IDEAL_CONSENT_VERSION`, `IDEAL_APP_VERSION`
- `IDEAL_STYLEGAN_REPO`, `IDEAL_STYLEGAN_NETWORK`, `IDEAL_PRIOR_PATH`
- `IDEAL_PERSONA_FEMALE_POOL`, `IDEAL_PERSONA_MALE_POOL`

`IDEAL_PRIOR_PATH`는 `{gender}`와 `{race}` placeholder를 유지한 template이어도 됩니다.
실행 스크립트는 필수 값, example placeholder, secret 길이, PostgreSQL URL, Python
package, config, checkpoint, 양쪽 prior와 pool 구조를 확인한 뒤 서버를 시작합니다.

## Start and verify

```bash
./scripts/run_gpu_backend.sh
```

시작 실패 메시지의 `ERROR: VARIABLE ...` 또는 경로를 먼저 수정합니다. Uvicorn이
실행된 뒤 GPU host에서 확인합니다.

```bash
set -a
source .env.gpu
set +a
curl --fail \
  -H "X-IDEAL-GATEWAY-SECRET: $GPU_GATEWAY_SECRET" \
  "http://127.0.0.1:${IDEAL_PORT:-8000}/internal/v1/health"
```

DB, CUDA, generator, prior/pool, OpenCLIP 검증이 모두 성공해야 `200`이 반환됩니다.
인증이 없는 요청과 잘못된 secret도 각각 확인합니다.

```bash
curl -i "http://127.0.0.1:${IDEAL_PORT:-8000}/internal/v1/health"
curl -i -H 'X-IDEAL-GATEWAY-SECRET: wrong' \
  "http://127.0.0.1:${IDEAL_PORT:-8000}/internal/v1/health"
```

각 응답은 `401`, `403`이어야 합니다.

## HTTPS and handback

연구실의 승인된 reverse proxy 또는 tunnel로 local worker를 HTTPS에 연결합니다. 모든
path, form body, cookie, redirect와 `X-IDEAL-GATEWAY-SECRET` header를 그대로
전달해야 하며 인증 없는 별도 public port를 열지 않습니다. 외부 HTTPS URL에서도
인증 health check가 성공하면 연구자에게 다음 한 줄을 보냅니다.

```text
GPU_BACKEND_URL=https://<approved-lab-endpoint>
```

secret은 URL과 분리된 기존 안전 채널로만 다룹니다.
