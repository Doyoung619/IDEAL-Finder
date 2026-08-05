# 이상형 탐지기 (Ideal Type Detector)

공식 NVIDIA StyleGAN2-ADA FFHQ 생성기와 Maximum-Variance Line Query(MVLQ)를
사용하는 로컬 인간대상 얼굴 선호 실험 프로토타입입니다.

## 핵심 구현

- **실제 얼굴 생성:** FFHQ 256×256 `G_ema`에서 W latent를 생성하고
  `G.synthesis(W)`로 얼굴을 합성합니다.
- **단일 알고리즘:** `references/latent_preference_MVLQ_beamer_muSigma.pdf`와
  `references/요약본.pdf`의 MVLQ만 실험 1에서 실행합니다.
- **12라운드:** 각 `M ∈ {2, 4, 8, 16}` 블록은 12라운드입니다.
- **동아시아 인상 하드 필터:** OpenCLIP 분류 결과가 `east_asian`이고
  confidence가 0.50 이상인 얼굴만 표시합니다. 이는 외형 기반 모델 판정이며
  실제 국적이나 민족을 판정하지 않습니다.
- **추가 하드 필터:** 얼굴 검출, 성인, 선호 대상 성별, 깨끗한 실사 인물사진,
  생성 품질 조건을 모두 통과해야 합니다. 필터는 후보 부족 시에도 완화하지
  않습니다.
- **선호도 기록:** 선택한 얼굴의 선호도 1–10점과 선택 난이도 1–7점을
  필수 저장합니다. 반응시간은 기록하지만 참가자 화면에는 표시하지 않습니다.

## MVLQ

PCA whitening 공간에서 초기 belief를 `N(μ0, I)`로 둡니다. 이를 원래 W
공간으로 옮기면 일반 Gaussian prior가 됩니다. 매 라운드에는 다음을 수행합니다.

1. 누적 M-way 선택 likelihood와 prior로 posterior MAP `ẑt`를 계산합니다.
2. MAP에서의 Laplace covariance `Σt`를 계산합니다.
3. `Σt`의 최대 고유값 방향 `vt`를 선택합니다.
4. `bm = -1 + 2(m-1)/(M-1)`로 두고 아래 M개 쿼리를 생성합니다.

```text
z_t^(m) = z_hat_t + sqrt(lambda_max(Sigma_0)) * b_m * v_t
```

5. 사용자가 고른 쿼리의 multinomial softmax likelihood를 누적하고 posterior를
   갱신합니다.

기본 쿼리 반지름은 문서대로 초기 covariance에서 자동 결정되며 사용자 조절
하이퍼파라미터는 없습니다. posterior MAP이 하드 필터의 feasible set 밖으로
나가면 직전 선택 얼굴을 constrained MAP 근사로 사용합니다. 전체 직선 중
하나라도 하드 필터를 통과하지 못하면 균등 간격 구조를 유지한 채 반지름 전체를
단계적으로 축소합니다. 실제 scale은 metadata의 `mvlq_query_scale`에 저장됩니다.

## 설치와 실행

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_dev.ps1
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. 연구자 화면은
`http://127.0.0.1:8000/admin`입니다.

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh
bash scripts/run_dev.sh
```

StyleGAN2 소스, FFHQ checkpoint, OpenCLIP weights, PCA cache, SQLite DB,
생성 이미지와 실험 출력은 Git에 포함하지 않습니다. `setup.ps1`/`setup.sh`가
`requirements.txt` 설치 후 필요한 모델과 cache를 자동으로 준비합니다.

## 주요 설정

`configs/default.yaml`:

```yaml
generator:
  mode: "stylegan2_ada"
  network_path: "models/stylegan2-ffhq-256x256.pkl"

experiment:
  m_list: [2, 4, 8, 16]
  rounds_per_m: 12

search:
  mode: "mvlq"
  pca_dimensions: 32

filters:
  east_asian_confidence_threshold: 0.50
```

## 검증

전체 자동 테스트:

```powershell
python -m pytest
```

실제 StyleGAN MVLQ 스트레스 테스트:

```powershell
python scripts\stress_test_stylegan.py --m 8 --rounds 12 --target-gender female
```

결과는 `outputs/stylegan_stress_mvlq/round_*/`에 저장됩니다.

## 데이터

- SQLite: `data/ideal_type_detector.sqlite3`
- 생성 이미지와 W latent: `cache/{participant}/{stage}/{condition}/{round}/`
- 참가자별 posterior 상태: `outputs/{participant}/blocks/*/strategy_state.pkl`
- PCA 모델: `data/models/`

CSV/ZIP 내보내기:

```powershell
python scripts\export_data.py
```
