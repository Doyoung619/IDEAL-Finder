# GPU gateway API contract

The public browser never calls the GPU host directly. Vercel forwards the
original method, path, query string, form body, cookie, and safe headers. Every
GPU request includes:

```http
X-IDEAL-GATEWAY-SECRET: <GPU_GATEWAY_SECRET>
```

The GPU worker returns `401` when the header is absent and `403` when it is
wrong. The secret must never be placed in HTML, JavaScript, a browser cookie,
or a query string.

## Versioned control endpoint

### `GET /internal/v1/health`

- Authentication: required shared-secret header
- Request body: none
- Expected timeout: 5 seconds
- Success: `200 application/json`

```json
{
  "status": "ok",
  "role": "gpu",
  "database": "ok",
  "cuda": true,
  "generator_loaded": true,
  "openclip_loaded": true,
  "artifacts": "ok"
}
```

Mock mode returns `cuda: false` and `artifacts: "mock"`. No URL, secret,
checkpoint filename, database credential, or filesystem path is returned.

## Browser workflow contract

The existing production-tested form routes remain the data-plane contract.
Keeping them unchanged avoids duplicating the Persona and Bayesian state
machines in a second API. Vercel relays these routes byte-for-byte and injects
authentication server-side.

| Operation | Method and path | Request | Success response |
|---|---|---|---|
| Landing/consent | `GET /`, `POST /consent` | HTML form: `consent=yes` | HTML or `303` |
| Participant facts | `POST /basic-info` | Existing form fields; no account fields | `303` |
| Persona profile | `POST /persona` | `persona::<category>` and optional `priority` fields | `303` |
| Persona query | `GET /persona/candidates` | Signed session cookie | HTML containing 8 image IDs/URLs |
| Persona selection | `POST /persona/candidates` | `batch_id`, `selected_image_id`, `action`, `reaction_time_sec` | `303` |
| Persona confirmation/session creation | `POST /persona/confirm` | `initial_rating`, `selection_confidence` | `303` plus updated signed cookie |
| Block creation | `POST /experiment1/start` | `block_index` | `303` |
| Query/resume | `GET /experiment1/round?block=<n>` | Signed cookie | Current deterministic round HTML; an existing query is reused |
| Selection/update | `POST /experiment1/round` | `block_index`, `round_id`, `selected_image_id`, `preference_rating`, `difficulty`, `reaction_time_sec` | `303` after durable selection and posterior snapshot |
| Session state/resume | Any existing study `GET` route | Signed cookie | Redirect to the next valid lifecycle screen |
| Generated image | `GET /generated/{image_id}.png` | Signed cookie | `200 image/png` from DB BLOB or recovered artifact |
| Final evaluation | `GET/POST /final-evaluation` | Existing evaluation form | HTML or `303` |
| Completion | `POST /survey` | `q1`…`q4`, optional `free_text` | `303 /complete` |

Responses use relative/same-origin URLs. GPU-local paths are never returned to
the browser. The gateway also rewrites an accidental absolute upstream
`Location` header to a same-origin path.

## Cookies

The GPU app owns and verifies the signed `session` cookie. It contains only
opaque participant/session identifiers, is `HttpOnly`, uses `SameSite=Lax`,
and is `Secure` whenever `APP_ENV=production`. Vercel only relays it and does
not need `SESSION_SECRET`.

## Timeouts and errors

- Gateway connect timeout: `GPU_CONNECT_TIMEOUT_SECONDS` (default 5 seconds)
- Gateway total upstream timeout: `GPU_REQUEST_TIMEOUT_SECONDS` (default 50 seconds)
- Gateway upstream timeout: 50 seconds (`GPU_REQUEST_TIMEOUT_SECONDS`)
- Missing/wrong GPU secret: `401` / `403`
- GPU unavailable: gateway `503` with `Retry-After: 3`
- GPU timeout: gateway `504` with `Retry-After: 3`
- DB unavailable: GPU `503`
- Invalid or duplicate form submission: existing redirect/idempotency flow

Selection data is committed before posterior mutation. A retry finds the
existing selection/round and will not apply the Bayesian update twice.

Real GPU timings are not yet available in this repository. Runtime logs now
emit `algorithm_latency_ms`, `generation_latency_ms`, and
`total_request_latency_ms`, tagged with algorithm, M, and round. Measure Persona
and M=2/4/8 on the lab host before public collection. If the observed worst
case exceeds the configured gateway ceiling, that is an operational decision
point; this change deliberately does not introduce a queue without evidence.
