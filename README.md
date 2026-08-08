# IDEAL Finder

StyleGAN2-ADA FFHQ 얼굴 latent에서 사용자의 반복 선택으로 이상형 preference point를 추정하는 연구 프로토타입입니다. 사용자에게 제공되는 query-selection 방식은 **Entropy Query** 하나뿐입니다.

Entropy Query는 현재 거리 기반 softmax preference model 아래에서 M-way mutual information을 직접 최적화하는 BACE-M style 적용입니다. 특정 논문의 결과를 그대로 재현한다고 주장하지 않으며, 이 프로젝트의 latent ideal-point likelihood에 맞춘 구현입니다.

## Interaction Model

현재 posterior를 $p_t(z)$, MAP을 $\hat z_t$, M개 latent query를 $Q=\{q^{(1)},\ldots,q^{(M)}\}$라고 합니다. preference parameter가 $z$일 때 m번째 이미지가 선택될 확률은 다음과 같습니다.

$$
\pi_m(z;Q)=\operatorname{softmax}_m\left(-\frac{1}{2}\lVert q^{(m)}-z\rVert_2^2\right)
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

이미지 bank에서 항목을 검색하지 않습니다. `core/entropy_query.py`가 latent space에서 M개 query point를 Adam multi-start optimization으로 직접 생성하고, 최적화가 끝난 좌표를 조건부 PCA prior를 통해 StyleGAN W-space로 옮긴 뒤 batch decode합니다. nearest-neighbor snapping이나 retrieval은 사용하지 않습니다.

Query가 과도하게 멀어지는 것을 막기 위해 초기 prior covariance $\Sigma_0=L_0L_0^T$로 정의되는 ellipsoid를 사용합니다.

$$
q^{(m)}=\hat z_t+L_0u^{(m)},\qquad \lVert u^{(m)}\rVert_2\le 1
$$

각 Adam step 후 모든 $u^{(m)}$을 unit ball로 projection합니다. covariance는 대칭화하고 Cholesky jitter를 $10^{-8}$부터 $10^{-3}$까지 증가시키며, 필요한 경우 eigenvalue를 clamp합니다. 모든 restart가 실패하면 다른 방식으로 fallback하지 않고 명시적인 `RuntimeError`를 발생시킵니다.

## Conditional Latent Prior

동일한 StyleGAN2-ADA FFHQ generator에서 얼굴을 생성하고 FairFace 확률을 operational filter로 사용해 조건별 W latent mean과 PCA를 구축합니다.

$$
w(\theta,c)=\mu_c+U_{c,d}\operatorname{diag}(\sqrt{\lambda_{c,1}},\ldots,\sqrt{\lambda_{c,d}})\theta
$$

기본 theta prior는 $\mathcal{N}(0,I_d)$이며 기본 dimension은 12입니다. `female/east_asian` prior 생성 예시는 다음과 같습니다.

```powershell
python scripts\build_demographic_prior.py `
  --stylegan-pkl models\stylegan2-ffhq-256x256.pkl `
  --fairface-weights models\fairface_res34.pth `
  --gender female --race east_asian --latent-dim 12 `
  --num-samples 50000 --min-accepted 2000 `
  --output artifacts\priors\female_east_asian_d12.npz --seed 42
```

남성 prior는 `--gender male`과 `artifacts\priors\male_east_asian_d12.npz`를 사용합니다. 실제 checkpoint 없는 CPU smoke build에는 `--mock`을 사용할 수 있습니다.

## Configuration

`configs/default.yaml`의 query 설정은 다음 하나로 고정됩니다.

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
```

수치 최적화 옵션은 preference model parameter가 아닙니다. `algorithm`에 `entropy` 이외의 값을 넣으면 `Only the entropy query algorithm is currently supported.` 오류를 발생시키며 자동 fallback하지 않습니다. 환경변수 기반 실제 모델 설정은 `configs/entropy.yaml`을 참고하세요.

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

웹 앱은 `http://127.0.0.1:8000`에서 실행됩니다. 각 $M\in\{2,4,8,16\}$ block은 12라운드이며 라운드 소요 시간은 참가자 화면에 표시하지 않습니다. 선택한 이미지는 preference 1, 나머지는 0으로 기록됩니다.

## Vercel Demo Deployment

Vercel 무료 플랜 배포는 실제 StyleGAN2-ADA/CLIP checkpoint를 서버리스 함수에 올리는 연구 실행 경로가 아니라, 앱 흐름 검증용 demo generator를 대상으로 합니다. Vercel에서는 `index.py`가 FastAPI 앱을 export하고, `requirements-vercel.txt`만 설치해 함수 번들에서 `torch`, `open_clip_torch`, checkpoint, run output을 제외합니다.

필수 환경변수는 다음과 같습니다.

```bash
IDEAL_SECRET_KEY="long-random-secret"
IDEAL_ADMIN_PASSWORD="separate-admin-password"
IDEAL_DATABASE_URL="postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE?sslmode=require"
IDEAL_DEMO=1
IDEAL_ARTIFACT_STORAGE=database
IDEAL_DB_POOL=serverless
```

Vercel에서는 generated image/latent/state artifact를 외부 DB에 함께 저장하고 요청 시 `/tmp` scratch directory로 복원합니다. `/admin`은 `IDEAL_ADMIN_PASSWORD`가 있으면 HTTP Basic 사용자명 `admin`으로 보호되며, Vercel에서 password가 없으면 노출되지 않습니다. 로컬 연구 실행에서는 기본값 그대로 filesystem artifact를 사용합니다.

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
```

## Limitations

- FairFace 출력은 객관적인 정체성 판정이 아니라 conditional prior 구축을 위한 operational filtering criterion입니다.
- Classifier bias와 label 정의가 latent distribution에 영향을 줄 수 있습니다.
- Gaussian/PCA prior는 실제 conditional latent manifold의 근사입니다.
- Monte Carlo mutual information은 sample 수에 따른 근사 오차가 있습니다.
- 생성 이미지는 실제 인물을 나타내지 않습니다.
- 사용자 실험 전에 StyleGAN, FFHQ, FairFace 및 기타 모델·데이터 라이선스를 별도로 검토해야 합니다.
- checkpoint, 생성 이미지, latent cache, prior artifact 및 run output은 Git에 포함하지 않습니다.
