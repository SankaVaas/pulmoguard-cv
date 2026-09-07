"""
Centralized application configuration.

All runtime configuration is environment-driven (12-factor style) via
pydantic-settings, with sane local-dev defaults. Nothing that varies
between environments (dev/staging/prod) is hardcoded in application code.

Secrets (JWT_SECRET_KEY, ADMIN_PASSWORD_HASH) MUST be overridden via
environment variables or a mounted secret in any non-local environment -
the defaults here are for local development only and are intentionally
weak/obvious so they are never mistaken for production-safe values.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App metadata ---
    APP_NAME: str = "PulmoGuard API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="development | staging | production")

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- CORS ---
    # Comma-separated origins in the env var, parsed into a list here.
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # --- Auth / JWT ---
    JWT_SECRET_KEY: str = Field(
        default="INSECURE-DEV-ONLY-CHANGE-ME",
        description="HMAC secret for signing JWTs. MUST be overridden in staging/production.",
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # --- Cookie-based session (browser clients) ---
    # The frontend receives its JWT as an httpOnly cookie rather than in a
    # JS-readable response body field, so it's never exposed to page
    # JavaScript (mitigates XSS token theft). API/CLI clients can still use
    # the bearer token returned in the /token response body directly - see
    # docs/ARCHITECTURE.md §5.3 for the full rationale.
    AUTH_COOKIE_NAME: str = "pulmoguard_access_token"

    # --- Bootstrap admin user ---
    # A single-user credential store is intentional for this project's scope
    # (an internal clinical-triage tool, not a multi-tenant SaaS). See
    # docs/ARCHITECTURE.md for the path to a real user store / IdP.
    ADMIN_USERNAME: str = "admin"
    # bcrypt hash of the default dev password "changeme" - override in real deployments.
    # Generate a new one with:
    #   python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode())"
    ADMIN_PASSWORD_HASH: str = "$2b$12$ek9El6/Oa92uA1/5udEX3eWw/OUc/0I4a5mRV5DmtMBsshtZW4csm"

    # --- Model / ML ---
    ML_CONFIG_PATH: str = "../ml/configs/config.yaml"
    ML_CHECKPOINT_PATH: str = "../ml/outputs/checkpoints/pulmoguard_best.pt"

    # --- Rate limiting ---
    # Applies to all routes by default via SlowAPIMiddleware; the auth
    # endpoint gets its own (stricter) override since brute-force login
    # attempts are the primary risk that rate limiting defends against here.
    RATE_LIMIT_PER_MINUTE: int = 30
    AUTH_RATE_LIMIT_PER_MINUTE: int = 10
    RATE_LIMIT_ENABLED: bool = True

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}, got '{v}'")
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton - environment is read once per process."""
    return Settings()
