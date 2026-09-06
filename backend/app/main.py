"""
PulmoGuard backend application entrypoint.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Run in production (see Dockerfile / docker-compose.yml):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import auth, health, predict
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.middleware.request_context import RequestContextMiddleware
from app.services.model_service import load_model

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle. Model loading happens here (not at import
    time) so a missing checkpoint fails the container's startup probe
    cleanly instead of crashing on the first request."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]")
    try:
        load_model()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        # Re-raise: an API server with no model is not a valid running state.
        # In Kubernetes this surfaces as CrashLoopBackOff with the log above
        # explaining exactly what's missing, rather than a silent 503 loop.
        raise
    yield
    logger.info("Shutting down PulmoGuard API.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Uncertainty-aware chest X-ray pneumonia triage API. Uses Monte Carlo "
        "Dropout to estimate predictive uncertainty and abstains on "
        "low-confidence cases rather than forcing a diagnosis."
    ),
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# --- Middleware (order matters: outermost added last runs first) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)


# --- Global exception handlers: never leak stack traces to clients ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request.", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"Unhandled exception on request {request_id}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error.", "request_id": request_id},
    )


# --- Routes ---
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", tags=["root"])
def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs" if settings.ENVIRONMENT != "production" else "disabled in production",
    }
