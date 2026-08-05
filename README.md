# IDEAL Finder

StyleGAN2-ADA FFHQ의 얼굴 latent에서 사용자의 이상형 선호를 반복 선택으로 추정하는 연구 프로토타입입니다. 기본 알고리즘은 Maximum-Variance Line Query(MVLQ)이며, 선택적으로 성별·인종 조건부 W-space prior와 demographic-constrained MVLQ를 사용할 수 있습니다.

## 핵심 구성

- **실제 얼굴 생성:** NVIDIA StyleGAN2-ADA PyTorch의 `G_ema`를 lazy-load하고 Z→W mapping과 W→image synthesis를 수행합니다. StyleGAN은 단순 스타일 변환 디코더가 아니라 FFHQ 얼굴로 학습된 생성 모델을 사용합니다.
- **조건부 prior:** 동일한 StyleGAN에서 얼굴을 생성하고 FairFace 확률로 operational filtering한 W latent만 사용해 조건별 평균과 PCA를 계산합니다.
- **Whitened theta:** 조건 $c$에서 저차원 좌표를 다음처럼 정의하므로 기본 prior는 $\theta \sim \mathcal{N}(0,I_d)$입니다.

  $$
  w(\theta,c)=\mu_c+U_{c,d}\operatorname{diag}(\sqrt{\lambda_{c,1}},\ldots,\sqrt{\lambda_{c,d}})\theta
  $$

- **Constrained MVLQ:** posterior covariance의 최대 분산 방향 $v_t$에서 균등 계수 $b_m=-1+2(m-1)/(M-1)$를 사용합니다.

  $$
  \theta_t^{(m)}=\hat\theta_t+\alpha_t r_t b_m v_t
  $$

  모든 후보가 성별·인종 threshold를 통과할 때까지 **query별 값이 아닌 하나의 공통 \(\alpha_t\)** 를 backtracking합니다. 따라서 후보들이 한 직선 위에 있고 균등 간격·대칭 구조가 유지됩니다. MAP이 불가능하면 posterior mean, prior mean, 고밀도 posterior center 순으로 대체하며 끝까지 실패하면 `NoFeasibleQuerySetError`를 발생시킵니다.
- **Bayesian update:** 기본 likelihood는 theta-space Euclidean distance 기반 M-way softmax입니다. `distance_space: w`를 선택하면 조건부 PCA가 유도한 W-space metric을 사용합니다.
- **기존 호환성:** 기존 unconstrained `mvlq`와 다른 전략 API는 그대로 유지됩니다.

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

웹 앱은 `http://127.0.0.1:8000`, 관리자 화면은 `http://127.0.0.1:8000/admin`에서 열립니다. 기본 실험은 각 $M\in\{2,4,8,16\}$마다 12라운드이며 참가자 화면에는 라운드 소요 시간을 표시하지 않습니다. 선택 이미지와 비선택 이미지의 선호도 표시는 각각 1과 0으로 기록됩니다.

## 외부 모델

다음 파일은 사용자가 별도로 준비해야 하며 Git에 포함하지 않습니다.

- NVIDIA StyleGAN2-ADA PyTorch source와 FFHQ `.pkl`
- FairFace ResNet-34 checkpoint `.pth`
- OpenCLIP weights와 프로젝트 runtime cache

기본 StyleGAN과 OpenCLIP 자산은 `scripts/download_models.py` 또는 setup script로 준비할 수 있습니다. FairFace weight는 해당 모델의 배포처와 라이선스를 확인한 뒤 경로를 CLI/config에 지정하세요.

## 조건부 Prior 생성

Female / East Asian:

```powershell
python scripts\build_demographic_prior.py `
  --stylegan-pkl models\stylegan2-ffhq-256x256.pkl `
  --fairface-weights models\fairface_res34.pth `
  --gender female --race east_asian --latent-dim 12 `
  --num-samples 50000 --min-accepted 2000 `
  --gender-threshold 0.90 --race-threshold 0.80 `
  --output artifacts\priors\female_east_asian_d12.npz --seed 42
```

Male / East Asian:

```powershell
python scripts\build_demographic_prior.py `
  --stylegan-pkl models\stylegan2-ffhq-256x256.pkl `
  --fairface-weights models\fairface_res34.pth `
  --gender male --race east_asian --latent-dim 12 `
  --num-samples 50000 --min-accepted 2000 `
  --output artifacts\priors\male_east_asian_d12.npz --seed 43
```

한·중·일을 포함한 극동아시아 기준에는 기본 `--race east_asian`을 사용합니다. 더 넓은 operational filter가 필요하면 `--race east_asian --race southeast_asian`처럼 복수 label을 합산할 수 있습니다. `--selection-mode soft`는 hard threshold 대신 성별 확률과 target race 확률 합의 곱을 sample weight로 사용합니다. 기본적으로 이미지를 저장하지 않으며 `--debug-grid generated\prior_debug.png`를 준 경우에만 작은 grid를 만듭니다. `--cache artifacts\latent_cache\bank.npz`로 안전한 NPZ latent bank를 재사용할 수 있습니다.

체크포인트 없는 CPU smoke build:

```powershell
python scripts\build_demographic_prior.py --mock --gender female `
  --latent-dim 4 --num-samples 256 --min-accepted 10 `
  --gender-threshold 0.70 --race-threshold 0.70 `
  --output artifacts\priors\mock.npz
```

## Constrained MVLQ Demo

```powershell
python scripts\run_conditional_mvlq_demo.py `
  --prior artifacts\priors\female_east_asian_d12.npz `
  --stylegan-pkl models\stylegan2-ffhq-256x256.pkl `
  --fairface-weights models\fairface_res34.pth `
  --num-options 5 --rounds 10 `
  --output outputs\demo_female_east_asian
```

각 라운드의 query grid, theta/W 좌표, 선택, posterior MAP, covariance eigenvalue, 공통 alpha, demographic score가 PNG와 JSON으로 저장됩니다. `--mock`을 사용하면 실제 checkpoint 없이 동일한 흐름을 실행할 수 있습니다.

웹 앱에서 constrained 전략을 사용하려면 `configs/default.yaml`의 `query.strategy`를 `demographic_constrained_mvlq`로 바꾸고 `conditional_prior.artifact_path`, `demographic.weights`를 지정합니다. 환경변수 기반 예시는 `configs/demographic_mvlq.yaml`에 있으며 필요한 변수가 없으면 설정 로딩 단계에서 변수명을 포함한 오류를 냅니다.

## 주요 API

- `StyleGAN2GeneratorAdapter.map_z_to_w(z)`, `synthesize_w(w)`, `generate_from_theta(theta, prior)`
- `FairFaceDemographicClassifier.predict_proba(images)`
- `ConditionalPCAPrior.sample_theta`, `theta_to_w`, `w_to_theta`, `log_prob`, `save`, `load`
- `DemographicPriorBuilder.build(config)`
- `DemographicConstrainedMVLQ.propose(...)`
- `GaussianPreferencePosterior.initialize_from_prior(prior)`와 `update(...)`

## 검증

```powershell
python -m pytest
python -m compileall -q app core scripts
python scripts\build_demographic_prior.py --help
python scripts\run_conditional_mvlq_demo.py --help
```

## 한계와 책임 있는 사용

- Demographic classifier 출력은 객관적인 정체성 판정이 아니라 이 프로토타입의 **operational filtering criterion**입니다.
- Classifier bias와 label 정의가 conditional prior 및 query distribution에 직접 영향을 줄 수 있습니다.
- Gaussian/PCA prior는 실제 conditional latent manifold의 근사입니다.
- Threshold가 너무 높으면 feasible query가 없거나 공통 query radius가 지나치게 작아질 수 있습니다.
- 생성 이미지는 실제 인물을 나타내지 않으며 개인정보나 실제 사용자 얼굴을 prior 구축에 사용하지 않습니다.
- 연구·사용자 실험 전에 StyleGAN, FFHQ, FairFace 및 기타 데이터·모델 라이선스를 별도로 검토해야 합니다.
- 모델 weight, 생성 이미지, 대용량 latent bank, prior artifact, run output 및 cache는 `.gitignore`로 제외됩니다.
