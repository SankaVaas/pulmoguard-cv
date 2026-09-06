"""
Health check routes.

Exposes separate liveness and readiness semantics, matching standard
Kubernetes probe conventions:
  - /health/live  : is the process up at all? (always 200 if the server can respond)
  - /health/ready : is the process ready to serve real traffic? (checks model is loaded)

Load balancers / orchestrators should route traffic based on /health/ready,
not /health/live - a process that's up but has no model loaded should be
taken out of rotation, not sent prediction requests that will 503.
"""
from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.schemas.prediction import HealthResponse
from app.services.model_service import is_model_loaded

router = APIRouter(prefix="/health", tags=["health"])
settings = get_settings()


@router.get("/live")
def liveness() -> dict:
    return {"status": "alive"}


@router.get("/ready", response_model=HealthResponse)
def readiness(response: Response) -> HealthResponse:
    model_loaded = is_model_loaded()
    if not model_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ready" if model_loaded else "not_ready",
        model_loaded=model_loaded,
        environment=settings.ENVIRONMENT,
        version=settings.APP_VERSION,
    )
