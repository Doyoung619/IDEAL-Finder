# 초기 실행 캐시

첫 실험 라운드가 느린 이유는 StyleGAN 로딩, latent PCA 준비, 그리고 동양인·성인·20대·성별 하드필터를 통과하는 후보군의 CLIP 평가가 첫 요청에 몰리기 때문입니다.

이미지나 후보 결과물은 Git에 넣지 않습니다. 대신 모델을 내려받은 뒤 아래 명령을 한 번 실행하면, 현재 컴퓨터에서만 재생성 가능한 캐시가 `data/precomputed/`에 만들어집니다.

```powershell
python scripts\prepare_initial_cache.py
```

이 명령은 다음을 수행합니다.

- 고정 seed로 latent PCA 준비
- `female`, `male` 각각 64개의 하드필터 통과 후보 생성
- 후보 latent와 필터 판정 metadata만 압축 저장
- 실제 얼굴 PNG는 저장하지 않음

앱은 `data/precomputed/initial_female_east_asian_20s.npz` 또는 `initial_male_east_asian_20s.npz`가 있으면 첫 라운드에서 후보를 다시 CLIP 평가하지 않고, 필요한 얼굴만 StyleGAN으로 즉시 복원합니다. 캐시가 없거나 모델/config가 바뀌면 기존 안전한 생성 경로로 자동 fallback합니다.

캐시는 로컬 실행 산출물이므로 Git에 커밋하지 않습니다. 캐시를 다시 만들려면 `data/precomputed/`의 해당 파일을 지우고 명령을 재실행하면 됩니다.
