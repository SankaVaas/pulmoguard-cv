# PulmoGuard — API Reference

Base URL (local): `http://localhost:8000`. Interactive Swagger docs are
available at `/docs` in any non-production environment (disabled in
production — see `ENVIRONMENT` in `backend/app/core/config.py`).

All endpoints return JSON. Errors follow the shape `{"detail": "..."}`
(FastAPI's default), except unhandled 500s, which also include a
`request_id` for log correlation — see `X-Request-ID` below.

---

## Authentication

### `POST /api/v1/auth/token`

Exchange credentials for a bearer token. Uses OAuth2 password-grant form
encoding (`application/x-www-form-urlencoded`), not JSON — this is what
lets Swagger UI's "Authorize" button and standard OAuth2 client libraries
work against this endpoint unmodified.

**Request body** (form-encoded):
| Field | Type | Required |
|---|---|---|
| `username` | string | yes |
| `password` | string | yes |

**Response `200`:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in_minutes": 60
}
```

**Response `401`:** incorrect username or password.

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme"
```

---

## Prediction

### `POST /api/v1/predict`

Upload a chest X-ray image and receive a triage prediction with an
uncertainty-based abstention decision. **Requires** a valid bearer token.

**Headers:**
| Header | Value |
|---|---|
| `Authorization` | `Bearer <token>` |

**Request body** (`multipart/form-data`):
| Field | Type | Notes |
|---|---|---|
| `file` | file | JPEG, PNG, or BMP. Max 10 MB. |

**Response `200`:**
```json
{
  "predicted_class": "PNEUMONIA",
  "confidence": 0.94,
  "normalized_entropy": 0.09,
  "abstain": false,
  "class_probabilities": { "NORMAL": 0.06, "PNEUMONIA": 0.94 },
  "mc_dropout_passes": 20,
  "filename": "xray.jpeg",
  "message": "Prediction confidence within accepted operating range."
}
```

| Field | Meaning |
|---|---|
| `predicted_class` | Argmax of the mean MC-Dropout probability distribution |
| `confidence` | Max mean probability across MC-Dropout passes |
| `normalized_entropy` | Predictive entropy scaled to `[0, 1]` — higher means less certain |
| `abstain` | `true` if `normalized_entropy` exceeds the configured threshold; the system recommends clinician review instead of trusting `predicted_class` |
| `class_probabilities` | Full distribution over all classes |

**Error responses:**
| Status | Cause |
|---|---|
| `400` | Unsupported content type, or empty file |
| `401` | Missing, invalid, or expired token |
| `413` | File exceeds 10 MB |
| `500` | Inference failure (see server logs, correlate via `request_id`) |

**Example:**
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=changeme" | jq -r .access_token)

curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sample_xray.jpeg"
```

---

## Health

### `GET /health/live`

Liveness probe — returns `200` if the process can respond at all. Does
**not** check whether the model is loaded. Use for Kubernetes `livenessProbe`.

```json
{ "status": "alive" }
```

### `GET /health/ready`

Readiness probe — returns `200` only if the model checkpoint has loaded
successfully. Returns `503` otherwise. Use for Kubernetes `readinessProbe`
and load-balancer health checks; traffic should be routed based on this
endpoint, not `/health/live`.

```json
{
  "status": "ready",
  "model_loaded": true,
  "environment": "production",
  "version": "1.0.0"
}
```

---

## Cross-cutting behavior

- **CORS:** allowed origins are configured via `CORS_ORIGINS` (comma-separated) in backend config; only `GET`/`POST` are permitted.
- **Request tracing:** every response includes an `X-Request-ID` header. Supply your own to correlate a client-side error with server logs, or read the server-generated one back.
- **Rate limiting:** `RATE_LIMIT_PER_MINUTE` is defined in configuration as a forward-looking setting; enforcement is not yet wired into middleware — see `docs/ARCHITECTURE.md` for planned hardening.
