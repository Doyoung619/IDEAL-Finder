# IDEAL-Finder deployment checklist

## Vercel gateway fast path

Vercel CLI가 설치·로그인·프로젝트 연결을 확인하고, GPU endpoint를 먼저 검증한
뒤 gateway 전용 Production 변수와 배포를 처리하도록 다음 helper를 사용합니다.

```bash
./scripts/deploy_vercel_gateway.sh --check
./scripts/deploy_vercel_gateway.sh
```

두 번째 명령에서 직접 입력하는 핵심 값은 `GPU_BACKEND_URL`과
`GPU_GATEWAY_SECRET`뿐입니다. secret은 화면에 표시되지 않고 stdin으로 Vercel에
전달되며 shell history나 local env 파일에 기록되지 않습니다. 프로젝트가 아직
연결되지 않았다면 한 번만 개인 Hobby scope와 project를 선택합니다. 실제 upload 전
release commit/config hash와 `DEPLOY` 확인을 요구합니다.

이 helper는 `APP_ROLE`, `APP_ENV`, gateway timeout을 안전한 고정값으로 등록하지만
`IDEAL_DATABASE_URL`, `SESSION_SECRET`, `IDEAL_ADMIN_PASSWORD`는 Vercel로 보내지
않습니다. 공식 명령 근거는 [Vercel environment CLI](https://vercel.com/docs/cli/env),
[project linking](https://vercel.com/docs/cli/link),
[production deploy](https://vercel.com/docs/cli/deploy)를 참고하세요.

## Full checklist

1. [ ] Create managed PostgreSQL and copy its connection string.

   ```bash
   export IDEAL_DATABASE_URL='postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require'
   ```

2. [ ] Create the schema (new DB) or back up and apply the checked-in migration
   (existing DB).

   ```bash
   python scripts/init_database.py --config configs/persona_study.yaml
   # Existing DB only, after backup:
   psql "$IDEAL_DATABASE_URL" -v ON_ERROR_STOP=1 \
     -f migrations/001_experiment_persistence.sql
   psql "$IDEAL_DATABASE_URL" -v ON_ERROR_STOP=1 \
     -f migrations/002_expand_seed_columns.sql
   ```

3. [ ] Generate three different credentials; store none in Git.

   ```bash
   python -c 'import secrets; print(secrets.token_urlsafe(48))'
   python -c 'import secrets; print(secrets.token_urlsafe(48))'
   python -c 'import secrets; print(secrets.token_urlsafe(32))'
   ```

   Use them as `GPU_GATEWAY_SECRET`, `SESSION_SECRET`, and
   `IDEAL_ADMIN_PASSWORD`. Record the approved `IDEAL_CONSENT_VERSION` and
   release commit as `IDEAL_APP_VERSION`.

4. [ ] Give the lab operator the release commit,
   `docs/GPU_SERVER_HANDOFF.md`, DB URL, three server credentials,
   consent/app versions, and approved artifact locations through a secure
   channel. The researcher owns DB creation and migration; the operator does
   not create a separate database.

5. [ ] Receive and verify the lab HTTPS URL.

   ```bash
   export GPU_BACKEND_URL='https://<approved-lab-endpoint>'
   curl --fail -H "X-IDEAL-GATEWAY-SECRET: $GPU_GATEWAY_SECRET" \
     "$GPU_BACKEND_URL/internal/v1/health"
   ```

6. [ ] Add only the gateway variables to Vercel.

   ```bash
   vercel env add APP_ROLE production
   vercel env add APP_ENV production
   vercel env add GPU_BACKEND_URL production
   vercel env add GPU_GATEWAY_SECRET production --sensitive
   vercel env add GPU_REQUEST_TIMEOUT_SECONDS production
   vercel env add GPU_CONNECT_TIMEOUT_SECONDS production
   ```

   Do not add `IDEAL_DATABASE_URL` or `SESSION_SECRET` to Vercel; those belong
   only to the GPU process in Option A.

7. [ ] Deploy the public gateway and check both layers.

   ```bash
   vercel --prod
   curl --fail https://<vercel-domain>/api/health
   ```

8. [ ] Run one non-study Persona test session through the Vercel URL, including
   initial candidates, the one allowed “다른 얼굴 8개 보기” reroll, the 1–10
   initial alignment rating, a round selection, and browser refresh/resume.

9. [ ] Confirm PostgreSQL persistence and researcher status.

   ```bash
   IDEAL_DATABASE_URL="$IDEAL_DATABASE_URL" \
     python scripts/experiment_status.py --config configs/persona_study.yaml
   ```

10. [ ] Export and inspect the analysis ZIP before public collection.

    ```bash
    IDEAL_DATABASE_URL="$IDEAL_DATABASE_URL" \
      python scripts/export_experiment_data.py \
      --config configs/persona_study.yaml \
      --output outputs/exports/study.zip
    unzip -l outputs/exports/study.zip
    ```

11. [ ] Freeze the deployment to the recorded release commit and config hash.
    Confirm the GPU worker reports the same `IDEAL_APP_VERSION`, then start the
    study without further code/config changes.

For a GPU-free local rehearsal, run `./scripts/run_local_fullstack.sh` and open
`http://127.0.0.1:3000`.
