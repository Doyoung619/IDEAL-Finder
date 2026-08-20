# Persona study: first-time setup

이 문서는 `configs/persona_study.yaml`의 실제 StyleGAN3/OpenCLIP/FairFace 설정으로 연구 웹앱을 처음 실행하는 절차입니다. CPU demo는 아래의 "빠른 동작 확인"만 따르면 되지만, 실제 연구 모드는 NVIDIA GPU와 별도 checkpoint/artifact가 필요합니다.

## 1. 저장소와 Python 환경

기준 환경은 Linux, Python 3.10, CUDA가 동작하는 NVIDIA GPU입니다. 기본 연구 설정은 온라인 생성/CLIP에 `cuda:0`, offline FairFace 분류에 `cuda:1,2,3`을 사용합니다. GPU가 적으면 offline builder 실행 시 `demographic.devices`를 실제 장치에 맞게 별도 설정하되, 고정된 user-study config는 수집 직전에 수정하지 마세요.

```bash
git clone https://github.com/Doyoung619/IDEAL-Finder.git
cd IDEAL-Finder

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 실제 연구에 필요한 외부 파일

checkpoint와 생성된 pool은 크기 및 라이선스 문제로 Git에 포함하지 않습니다.
다음 파일을 준비한 뒤 `.env.gpu`의 환경변수에 현재 머신의 경로를
지정합니다. 연구 config나 Python 코드를 서버별 절대 경로로 수정하지 않습니다.

| 용도 | 환경변수 | 값 형식 |
|---|---|---|
| StyleGAN3 소스 | `IDEAL_STYLEGAN_REPO` | StyleGAN3 checkout directory |
| FFHQ StyleGAN3 pickle | `IDEAL_STYLEGAN_NETWORK` | checkpoint file |
| FairFace race weights | `IDEAL_FAIRFACE_RACE_WEIGHTS` | weights directory/file |
| FairFace gender weights | `IDEAL_FAIRFACE_GENDER_WEIGHTS` | weights directory/file |
| FairFace age weights | `IDEAL_FAIRFACE_AGE_WEIGHTS` | weights directory/file |
| YuNet face detector | `IDEAL_FACE_DETECTOR_PATH` | ONNX file |

아래 연구 artifact도 필요합니다.

```text
artifacts/priors/female_20_29_east_asian_d12.npz
artifacts/priors/male_20_29_east_asian_d12.npz
artifacts/persona_pools/female_20_29_east_asian_d12/pool.npz
artifacts/persona_pools/female_20_29_east_asian_d12/metadata.json
artifacts/persona_pools/male_20_29_east_asian_d12/pool.npz
artifacts/persona_pools/male_20_29_east_asian_d12/metadata.json
```

기존 연구 서버의 artifact를 재사용할 수 있으면 디렉터리 구조를 보존해 복사하는 것이 가장 빠릅니다. 복사본이 없다면 3–4절의 builder로 생성하세요.

## 3. 성별 + 20–29세 + East Asian 12D prior 생성

최종 연구 설정의 FairFace 성별, 20–29, East Asian operational threshold를
적용해 여성과 남성 prior를 각각 생성합니다. 이 label은 참가자의 정체성을
판정하는 용도로 사용하지 않습니다.

```bash
source .venv/bin/activate

python scripts/build_gender_age_prior.py \
  --config configs/persona_study.yaml \
  --gender female --accepted 2500 \
  --output artifacts/priors/female_20_29_east_asian_d12.npz

python scripts/build_gender_age_prior.py \
  --config configs/persona_study.yaml \
  --gender male --accepted 2500 \
  --output artifacts/priors/male_20_29_east_asian_d12.npz
```

## 4. Persona pool 생성 및 검증

실제 웹앱은 성별별 최소 1,000개의 검증된 pool item을 요구합니다. Builder는 checkpoint를 남기므로 같은 명령으로 재실행하면 이어서 진행합니다.

```bash
mkdir -p logs

nohup .venv/bin/python scripts/build_persona_pool.py \
  --config configs/persona_study.yaml \
  --gender female --target-size 1000 \
  --output artifacts/persona_pools/female_20_29_east_asian_d12 \
  > logs/female_pool.log 2>&1 &

nohup .venv/bin/python scripts/build_persona_pool.py \
  --config configs/persona_study.yaml \
  --gender male --target-size 1000 \
  --output artifacts/persona_pools/male_20_29_east_asian_d12 \
  > logs/male_pool.log 2>&1 &
```

완료 후 반드시 검증합니다.

```bash
.venv/bin/python scripts/validate_persona_pool.py \
  artifacts/persona_pools/female_20_29_east_asian_d12 --minimum-size 1000
.venv/bin/python scripts/validate_persona_pool.py \
  artifacts/persona_pools/male_20_29_east_asian_d12 --minimum-size 1000
```

## 5. 실행 전 검사

```bash
source .venv/bin/activate
python -m pytest
python -m compileall -q app core scripts
```

정상 기준은 테스트 전체 통과, 두 pool 검증 성공입니다. 현재 구현의 연구 schedule은 다음 조건을 만족합니다.

- M=2, 4, 8이 각각 두 번씩 등장하는 총 6블록
- 참가자별 M 순서 균형 랜덤화
- 각 M 안에서 Entropy/RC-MLQ 선후 균형 랜덤화
- 블록당 12라운드
- Entropy 최종 query offset 1.4배 확대
- 초기 posterior: 55% persona Gaussian + 45% global prior

## 6. 실제 연구 서버 실행

`run_persona_study.sh`는 두 pool을 다시 검증하고, 호환되지 않는 미완료 세션을 표시한 후 `0.0.0.0:8000`에서 서버를 시작합니다.

```bash
cd /path/to/IDEAL-Finder
bash scripts/run_persona_study.sh
```

다른 포트를 사용하려면:

```bash
IDEAL_PORT=18000 bash scripts/run_persona_study.sh
```

서버 자체에서 다음 명령이 `200`을 반환하면 정상입니다.

```bash
curl -o /dev/null -s -w '%{http_code}\n' http://127.0.0.1:8000/
curl -o /dev/null -s -w '%{http_code}\n' http://127.0.0.1:8000/admin
```

## 7. 로컬 컴퓨터에서 원격 서버 접속

서버의 8000번 포트가 외부 방화벽에 막힌 경우, 로컬 컴퓨터에서 기존에 사용하는 SSH 계정/호스트로 port forwarding을 실행합니다.

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@SERVER
```

그 터미널을 열어 둔 상태로 브라우저에서 `http://127.0.0.1:8000`에 접속합니다. 로컬 8000번이 사용 중이면 `-L 18000:127.0.0.1:8000`으로 바꾸고 `http://127.0.0.1:18000`을 여세요.

임시 공개 터널은 접근 URL을 아는 누구나 앱에 들어올 수 있으므로 실제 개인정보를 받는 본 실험에는 인증된 SSH/VPN 또는 기관 reverse proxy를 권장합니다.

## 8. 빠른 동작 확인(CPU demo)

실제 checkpoint와 pool을 준비하기 전 UI/API만 확인하려면:

```bash
source .venv/bin/activate
bash scripts/run_demo.sh
```

이 모드는 mock generator/embedding을 사용하므로 연구 결과로 사용하면 안 됩니다.

## 9. 데이터와 운영 주의사항

- SQLite DB 기본 위치: `data/ideal_type_detector.sqlite3`
- 참가자별 artifact: `outputs/<participant_id>/`
- 관리자 화면: `/admin`
- 내보내기: `/admin/export.zip`
- `data/`, `outputs/`, checkpoint, prior, persona pool은 Git에 올라가지 않습니다.
- `app.secret_key`는 실제 배포 전 반드시 예측 불가능한 값으로 변경하고 저장소에 커밋하지 마세요.
- 진행 중인 연구에서 `search.version` 또는 schedule을 바꾸면 기존 세션과 새 세션을 섞지 마세요.
