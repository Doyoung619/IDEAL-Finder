# IDEAL Finder

StyleGAN FFHQ 얼굴 latent에서 persona 기반 초기 얼굴을 고른 뒤 사용자의 반복 선택으로 이상형 preference point를 추정하는 연구 프로토타입입니다. 실험은 **Entropy Query**와 논문의 **RC-MLQ**를 같은 초기분포에서 비교합니다.

최종 user-study 운영자는 먼저 [`docs/STUDY_RELEASE.md`](docs/STUDY_RELEASE.md)를 확인하세요. GPU 서버 담당자용 절차는 [`docs/GPU_SERVER_HANDOFF.md`](docs/GPU_SERVER_HANDOFF.md), 연구자 배포 순서는 [`docs/DEPLOYMENT_CHECKLIST.md`](docs/DEPLOYMENT_CHECKLIST.md), gateway 계약은 [`docs/GPU_API_CONTRACT.md`](docs/GPU_API_CONTRACT.md), 알고리즘 고정 근거는 [`docs/ALGORITHM_AUDIT.md`](docs/ALGORITHM_AUDIT.md)에 있습니다. 모델·artifact 구축을 포함한 일반 설정은 [SETUP.md](SETUP.md)를 참고하세요.

GPU HTTPS URL과 shared secret을 받은 뒤 Vercel gateway는 다음 한 명령으로
설정·검증·배포할 수 있습니다. secret은 hidden prompt로 입력합니다.

```bash
./scripts/deploy_vercel_gateway.sh
```

## 처음 실행하기

실제 StyleGAN3 연구 환경의 checkpoint 배치, prior/pool 생성, 검증, 서버 실행 및 원격 접속 절차는 [SETUP.md](SETUP.md)에 순서대로 정리되어 있습니다.

이미 현재 연구 서버의 checkpoint와 artifact가 준비돼 있다면 다음 두 명령으로 시작할 수 있습니다.

```bash
python -m pytest
bash scripts/run_persona_study.sh
```

초기 posterior는 persona 선택 하나에 과도하게 쏠리지 않도록 다음 mixture로 구성합니다.

$$
p_0(z)=0.55\,\mathcal N(\mu_{persona},0.35\Sigma)+0.45\,p_{global}(z),
\qquad p_{global}=\mathcal N(0,\Sigma).
$$

8192 weighted particles를 사용하며 두 알고리즘은 같은 seed와 같은 초기 particle/weight를 받습니다.

## Interaction Model

현재 posterior를 $p_t(z)$, MAP을 $\hat z_t$, M개 latent query를 $Q=\{q^{(1)},\ldots,q^{(M)}\}$라고 합니다. preference parameter가 $z$일 때 m번째 이미지가 선택될 확률은 다음과 같습니다.

$$
\pi_m(z;Q)=\operatorname{softmax}_m\left(-\beta_t\lVert q^{(m)}-z\rVert_2^2\right)
$$

현재 posterior에서 reparameterized Monte Carlo sample을 뽑아 predictive entropy와 expected conditional entropy의 차이를 최대화합니다.

$$
\operatorname{MI}_t(Q)
=H\left(\mathbb{E}_{Z\sim p_t}[\pi(Z;Q)]\right)
-\mathbb{E}_{Z\sim p_t}\left[H(\pi(Z;Q))\right]
$$

$$
Q_t=\arg\max_Q\operatorname{MI}_t(Q)
$$

전체 interaction은 다음 순서로 동작합니다.

```text
posterior p_t
-> optimize M latent query points
-> decode with D(.)
-> observe Y_t
-> Bayesian update
-> compute MAP z_hat_(t+1)
```

## Direct Synthetic Queries

반복 refinement 단계에서는 이미지 bank에서 항목을 검색하지 않습니다. `core/entropy_query.py`가 latent space에서 M개 query point를 Adam multi-start optimization으로 직접 생성하고, 최적화가 끝난 좌표를 조건부 PCA prior를 통해 StyleGAN W-space로 옮긴 뒤 batch decode합니다. nearest-neighbor snapping이나 retrieval은 이 단계에서 사용하지 않습니다. Persona 초기 얼굴을 고르는 첫 단계만 별도의 offline pool retrieval을 사용합니다.

Query가 과도하게 멀어지는 것을 막기 위해 초기 prior covariance $\Sigma_0=L_0L_0^T$로 정의되는 ellipsoid를 사용합니다.

$$
q^{(m)}=\hat z_t+L_0u^{(m)},\qquad \lVert u^{(m)}\rVert_2\le 1
$$

각 Adam step 후 모든 $u^{(m)}$을 unit ball로 projection합니다. covariance는 대칭화하고 Cholesky jitter를 $10^{-8}$부터 $10^{-3}$까지 증가시키며, 필요한 경우 eigenvalue를 clamp합니다. 모든 restart가 실패하면 다른 방식으로 fallback하지 않고 명시적인 `RuntimeError`를 발생시킵니다.

최종 Entropy query는 posterior 중심으로부터의 offset을 `1.4`배 확대합니다. 따라서 최적화된 상대 배치는 유지하면서 모든 pairwise 거리가 정확히 1.4배 커지며, 확대 후 mutual information과 실제 최소 거리를 다시 계산해 artifact에 기록합니다. 이 배율은 `query.entropy_output_spread_scale`로 조정할 수 있습니다.

## Conditional Latent Prior

동일한 StyleGAN3 FFHQ generator에서 얼굴을 생성하고 FairFace 성별, 20–29, East Asian 확률을 operational filter로 사용해 조건별 W latent mean과 PCA를 구축합니다. 이 분류값은 참가자의 정체성을 판정하는 용도가 아니라 연구용 조건부 분포를 만드는 필터입니다.

$$
w(\theta,c)=\mu_c+U_{c,d}\operatorname{diag}(\sqrt{\lambda_{c,1}},\ldots,\sqrt{\lambda_{c,d}})\theta
$$

기본 theta prior는 $\mathcal{N}(0,I_d)$이며 기본 dimension은 12입니다. 여성 20–29 prior 생성 예시는 다음과 같습니다.

```bash
python scripts/build_gender_age_prior.py \
  --config configs/persona_study.yaml \
  --gender female --accepted 2500 \
  --output artifacts/priors/female_20_29_east_asian_d12.npz
```

남성 prior는 `--gender male`과 `artifacts/priors/male_20_29_east_asian_d12.npz`를 사용합니다.

## Persona warm start

새 참가자의 전체 상태 흐름은 다음과 같습니다.

```text
consent → basic_info → persona questionnaire → 8 persona candidates
→ confirmation → Entropy Query refinement → final evaluation → survey
```

Persona 문항은 `persona_v1`로 버전 관리됩니다. 최종 연구 설정의 생성 대상은 East Asian 조건을 통과한 한 명의 20–29세 성인, 정면 또는 거의 정면인 실사형 초상이며 기본 정보에서는 여성/남성 target을 고릅니다. 응답은 raw JSON, 한국어 요약, 영어 `base/full/category` prompt bundle로 함께 저장됩니다.

후보는 global W sample을 사후 projection하지 않습니다. 성별 conditional 12D prior에서 직접

$$
\theta_i\sim\mathcal N(0,I_{12}),\qquad w_i=T_c(\theta_i),\qquad x_i=G(w_i)
$$

순서로 미리 생성하므로 선택 이미지의 정확한 $\theta_i$와 $w_i$가 보존됩니다. OpenCLIP semantic score의 고정 가중치는 base 0.25, full 0.40, category 0.35이고, semantic top-64에서 $\lambda=0.75$ MMR로 8개를 보여줍니다. 이미 본 pool ID는 다음 페이지에서 제외됩니다.

확정한 $\theta_{\mathrm{persona}}$는 모든 block의 독립 mixture posterior에 동일하게 주입됩니다.

```text
p0 = 0.55 N(theta_persona, 0.35 Sigma) + 0.45 N(0, Sigma)
particles = 8192
history = []
```

두 알고리즘은 같은 seed로 생성한 동일한 초기 particle/weight를 받습니다. ESS가 particle 수의 12% 아래로 내려가면 deterministic Liu–West rejuvenation을 수행해 후반 particle collapse를 방지합니다.

## Persona pool 구축

Research pool은 성별별 1,000개 미만이면 웹 retrieval을 시작하지 않습니다. 정확히 한 얼굴, 품질, target gender, FairFace 20–29와 East Asian threshold를 통과한 항목만 저장합니다. Threshold는 실행 중 자동 완화되지 않습니다.

Linux/macOS background 실행:

```bash
mkdir -p logs
nohup python scripts/build_persona_pool.py \
  --gender female --target-size 1000 \
  --output artifacts/persona_pools/female_20_29_east_asian_d12 \
  > logs/female_pool.log 2>&1 &

nohup python scripts/build_persona_pool.py \
  --gender male --target-size 1000 \
  --output artifacts/persona_pools/male_20_29_east_asian_d12 \
  > logs/male_pool.log 2>&1 &
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force logs | Out-Null
Start-Process python `
  -ArgumentList "scripts/build_persona_pool.py --gender female --target-size 1000 --output artifacts/persona_pools/female_20_29_east_asian_d12" `
  -RedirectStandardOutput "logs/female_pool.log" `
  -RedirectStandardError "logs/female_pool.err.log"
```

Builder는 `pool_checkpoint.npz`와 config hash로 같은 실행만 resume하며 `progress.json`과 `progress.png`를 실제 처리량으로 매 batch 갱신합니다. 결과는 다음 명령으로 검사합니다.

```bash
python scripts/validate_persona_pool.py artifacts/persona_pools/female_20_29_east_asian_d12 --minimum-size 1000
```

`persona.require_real_clip: true`인 research mode에서는 OpenCLIP 또는 checkpoint 로드 실패 시 즉시 종료합니다. latent-hash/mock embedding은 `IDEAL_DEMO=1`, unit test, 또는 명시적인 builder `--mock`에서만 허용되며 scientific CLIP result로 기록되지 않습니다.

## Configuration

`configs/default.yaml`의 핵심 실험 설정은 다음과 같습니다.

```yaml
query:
  algorithm: entropy
  num_options: 5
  posterior_mc_samples: 512
  num_restarts: 8
  optimization_steps: 200
  learning_rate: 0.05
  constraint: prior_ellipsoid
  seed: 0
  beta: 1.0
  rc_resolution_min: 0.2
  rc_resolution_max: 6.0
  rc_resolution_steps: 120

persona:
  prior_covariance_scale: 0.35
  global_mixture_weight: 0.45
  mixture_particle_count: 8192
```

RC-MLQ는 posterior covariance의 최대 고유벡터로 방향을 정하고, projected posterior quantile로 slate 모양을 만든 뒤 posterior 폭과 독립된 고정 물리 grid `r∈[0.2,6.0]`에서 exact EIG를 최대화합니다. adaptive-beta 기능은 일반 설정에서 사용할 수 있지만, 최종 user-study 설정은 audit 결과에 따라 `beta: 1.0`, `adaptive_beta_enabled: false`로 고정합니다. 선택 난이도와 이상형 근접도 응답은 분석을 위해 계속 저장되며 다음 라운드의 beta를 바꾸지 않습니다.

Persona-only 연구 설정은 별도 파일로 제공됩니다. 참가자별로 `M=8,4,2` 순서를 균형 랜덤화하고, 각 M 안에서도 `Entropy/RC-MLQ` 선후를 독립적으로 균형 랜덤화해 총 6개 블록을 실행합니다. 각 블록은 12라운드입니다. 기존 answer-key와 recommendation-condition 순회는 끕니다. 실제 StyleGAN3, East Asian 조건의 성별별 20–29세 conditional prior, FairFace ViT, YuNet, OpenAI CLIP을 사용합니다.

```bash
bash scripts/run_persona_study.sh
```

```powershell
$env:IDEAL_CONFIG = "configs/persona_study.yaml"
python -m uvicorn app.main:app
```

## 설치 및 웹 실행

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh
bash scripts/run_dev.sh
```

웹 앱은 `http://127.0.0.1:8000`에서 실행됩니다. 실제 persona 연구에는 위 `configs/persona_study.yaml`을 사용하세요. 블록 순서는 participant ID와 study seed로 재현 가능한 균형 랜덤 schedule이며, 라운드 소요 시간은 참가자 화면에 표시하지 않습니다.

## Production data collection

연구 운영은 Vercel 함수에 StyleGAN/Torch/CUDA를 올리지 않고, 실제 모델과 기존 FastAPI 연구 로직을 GPU 서버의 단일 worker에 둡니다. PostgreSQL이 canonical storage이며 CSV는 분석용 export입니다.

```text
Participant browser
        |
        v
Vercel public gateway
        |
        v
GPU FastAPI (1 worker) ---- PostgreSQL
        |
        +---- StyleGAN3 / OpenCLIP / FairFace
```

GPU generation은 process 내 lock으로 직렬화됩니다. 따라서 운영 시 `uvicorn --workers 1`을 유지하고, 수평 확장이 필요하면 GPU worker/queue를 별도로 늘려야 합니다. mutable posterior, beta, round history, 이미지는 participant/block별 경로와 DB row로 분리됩니다.

### Database setup

새 PostgreSQL DB에는 app과 같은 Python 환경에서 현재 schema를 생성합니다.

```bash
export IDEAL_DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require'
python scripts/init_database.py --config configs/persona_study.yaml
```

기존 IDEAL-Finder DB를 upgrade할 때는 반드시 backup 후 versioned migration을 적용합니다. 기존 `selections` 중 `(block_id, round_id)` 중복이 있으면 migration이 중단되므로 연구자가 먼저 확인해야 합니다.

```bash
psql 'postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require' \
  -v ON_ERROR_STOP=1 \
  -f migrations/001_experiment_persistence.sql
```

새로 추가된 핵심 table은 다음과 같습니다.

- `experiment_sessions`: Persona 최종 확정 후에만 생성되는 전체 실험 단위
- `experiment_blocks`: 기존 6개 M×algorithm block과 session의 연결
- `experiment_rounds`: 선택, final query, rating, beta, posterior summary를 담는 라운드 단위 row
- `experiment_events`: session/block/query/choice/completion/recovery event

`latent_images.image_data`, `latent_data`, `experiment_blocks.strategy_state_data`는 production에서 정확히 본 PNG, W latent, 현재 particle posterior state를 PostgreSQL `BYTEA`로 보존합니다. 작은–중간 규모 실험에서 별도 object storage 의존성을 추가하지 않기 위한 최소 변경입니다. DB 크기가 운영 한계에 도달하면 이 컬럼만 object storage key로 교체하세요.

### Experiment lifecycle and recovery

Persona questionnaire와 8개 후보 선택만으로는 session이 생성되지 않습니다. Persona 얼굴 최종 확정 시 `experiment_sessions` row와 `session_created` event가 commit됩니다. 각 선택 POST는 다음 순서로 처리됩니다.

```text
server validates displayed image + expected round
-> Selection / FaceRating / ExperimentRound(status=submitted) COMMIT
-> existing Bayesian update runs unchanged
-> posterior snapshot + status=completed COMMIT
-> redirect to next round
```

첫 commit 후 posterior update가 실패하면 선택은 이미 DB에 남고 round는 `error`로 표시됩니다. 새로고침/재제출 시 state history 길이를 확인해 Bayesian update를 한 번만 적용합니다. `(block_id, round_id)`와 `(session_id, block_id, round_index)` unique constraint가 중복 저장을 막습니다. 서명된 session cookie가 opaque participant/session ID를 보관하므로 새로고침 후 completed round 다음으로 복구됩니다.

### Environment variables

split deployment에서는 Vercel용 `.env.gateway.example`과 GPU 서버용
`.env.gpu.example`을 사용하고 실제 secret이나 DB URL을 Git에 commit하지
마세요. 핵심 변수는 다음과 같습니다.

- `IDEAL_DATABASE_URL`: 일반 PostgreSQL connection string
- `SESSION_SECRET`: 참가자 session cookie 서명 key (`IDEAL_SECRET_KEY`는 호환 alias)
- `GPU_GATEWAY_SECRET`: Vercel→GPU 요청 전용 shared secret
- `GPU_BACKEND_URL`: Vercel에서만 사용하는 GPU HTTPS base URL
- `IDEAL_ADMIN_PASSWORD`: `/admin`/export Basic Auth password
- `IDEAL_CONFIG`: 연구 운영은 `configs/persona_study.yaml`
- `IDEAL_ARTIFACT_STORAGE=database`: exact image/latent/posterior state DB 보존
- `IDEAL_STYLEGAN_*`, `IDEAL_FAIRFACE_*`, `IDEAL_PRIOR_PATH`, `IDEAL_PERSONA_*`: GPU host artifact 경로 override
- `IDEAL_CONSENT_VERSION`, `IDEAL_APP_VERSION`: 세션에 남길 protocol/release 버전

### GPU server

GPU host에 repository, model, prior, Persona pool을 배치하고 `.env.gpu`의
경로를 해당 host에 맞게 설정한 뒤 실행합니다. 자세한 전달용 절차는
[`docs/GPU_SERVER_HANDOFF.md`](docs/GPU_SERVER_HANDOFF.md)에 있습니다.

```bash
cp .env.gpu.example .env.gpu
./scripts/run_gpu_backend.sh
```

프로세스는 인증된 `/internal/v1/health`에서 app/DB/CUDA/model 상태를
노출하며 secret이나 path는 반환하지 않습니다. GPU 서버를 재시작해도 DB
BLOB에서 block state와 generated artifact를 scratch path로 복원합니다.

### Status and CSV export

```bash
python scripts/experiment_status.py --config configs/persona_study.yaml
python scripts/export_experiment_data.py \
  --config configs/persona_study.yaml \
  --output outputs/exports/study.zip
```

ZIP에는 normalized raw table CSV와 라운드당 한 행인 `rounds_flat.csv`/`analysis_rounds.csv`가 함께 들어갑니다. export는 민감한 연구 자료로 취급하고 암호화된 저장소에 두세요.
DB의 binary image/latent/state 본문은 CSV에 복제하지 않고 byte size만 표시합니다.

### Vercel deployment boundary

`torch`, CUDA, StyleGAN3 checkpoint, OpenCLIP/FairFace는 Vercel에서 실행하지
않습니다. production Vercel은 신뢰할 수 있는 HTTPS GPU endpoint로만 요청을
전달하는 public gateway입니다. 기존 local/demo monolith 실행은
`app.main:app` entrypoint로 계속 지원합니다.

`index.py`는 이제 `APP_ROLE=gateway`를 기본으로 하며 `app.gateway`만 import합니다.
`requirements-vercel.txt`에는 FastAPI와 HTTP client만 있고, `.vercelignore`가
model/DB/research tree를 deployment upload와 Python function bundle에서
제외합니다. CSS/JavaScript는 gateway가
직접 제공하며, 동적 HTML/form은 같은 public origin에서 GPU 앱으로 전달합니다.
gateway는 브라우저의 기존
form/cookie/redirect를 그대로 GPU 앱으로 전달하고 secret header를 server-side로
추가합니다. API 세부 계약은
[`docs/GPU_API_CONTRACT.md`](docs/GPU_API_CONTRACT.md), 연구자 순서는
[`docs/DEPLOYMENT_CHECKLIST.md`](docs/DEPLOYMENT_CHECKLIST.md)를 참고하세요.

실제 GPU 없이 전체 경로를 확인하려면 다음 한 명령을 실행합니다.

```bash
./scripts/run_local_fullstack.sh
```

## Entropy Demo

실제 StyleGAN decoder:

```powershell
python scripts\run_entropy_demo.py `
  --query-algorithm entropy `
  --prior artifacts\priors\female_east_asian_d12.npz `
  --stylegan-pkl models\stylegan2-ffhq-256x256.pkl `
  --num-options 5 --rounds 2 `
  --output outputs\entropy_demo
```

CPU mock smoke:

```powershell
python scripts\run_entropy_demo.py --mock --query-algorithm entropy `
  --prior artifacts\priors\mock.npz --rounds 2 `
  --posterior-mc-samples 64 --num-restarts 2 --optimization-steps 30 `
  --output outputs\entropy_mock
```

각 `round_XX` 디렉터리에 `query_points.npy`, `query_center.npy`, `decoded_images/`, `posterior_mean.npy`, `posterior_covariance.npy`, `map_estimate.npy`, `entropy_metrics.json`, `observed_choice.json`을 저장합니다.

## Validation

```powershell
python -m pytest
python -m compileall -q app core scripts
python scripts\run_entropy_demo.py --help
python scripts\build_persona_pool.py --help
python scripts\validate_persona_pool.py --help
```

## Limitations

- FairFace 출력은 객관적인 정체성 판정이 아니라 conditional prior 구축을 위한 operational filtering criterion입니다.
- Classifier bias와 label 정의가 latent distribution에 영향을 줄 수 있습니다.
- Gaussian/PCA prior는 실제 conditional latent manifold의 근사입니다.
- Monte Carlo mutual information은 sample 수에 따른 근사 오차가 있습니다.
- 생성 이미지는 실제 인물을 나타내지 않습니다.
- Persona-only 연구에는 no-persona control이 없으므로 persona가 no-persona보다 우수하다는 인과 비교는 할 수 없습니다. 측정 대상은 warm start 이후 refinement의 변화입니다.
- OpenCLIP prompt와 tokenization, FairFace age/gender/race label, face detector에는 편향과 측정 오차가 있으며 participant-facing 정체성 판정으로 사용하면 안 됩니다.
- 사용자 실험 전에 StyleGAN, FFHQ, FairFace 및 기타 모델·데이터 라이선스를 별도로 검토해야 합니다.
- checkpoint, 생성 이미지, latent cache, prior artifact 및 run output은 Git에 포함하지 않습니다.
