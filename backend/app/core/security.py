"""
Authentication primitives: password verification and JWT issuance/decoding.

Design notes:
  - Password hashing uses the `bcrypt` library directly (not passlib) -
    passlib's bcrypt backend has known incompatibilities with recent
    bcrypt releases (missing `__about__`), so we avoid that dependency
    entirely and call bcrypt's straightforward hashpw/checkpw API.
  - JWTs are signed with HS256 using a symmetric secret (JWT_SECRET_KEY).
    For a multi-service deployment, migrate to asymmetric signing (RS256)
    so only the auth service holds the private key - see docs/ARCHITECTURE.md.
  - Tokens carry a `sub` (subject/username) claim and an `exp` claim.
    No roles/scopes are modeled yet since this is a single-admin-user
    system by design (see core/config.py); the structure is ready to
    extend with a `scopes` claim if multi-role auth is needed later.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings

settings = get_settings()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash. Returns False
    (never raises) on malformed hashes so callers can treat this as a
    simple boolean auth check."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Issue a signed JWT for the given subject (username)."""
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": subject, "exp": expire, "iat": datetime.now(UTC)}
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Decode and validate a JWT, returning the subject (username) if valid,
    or None if the token is expired, malformed, or has an invalid signature."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def authenticate_admin(username: str, password: str) -> bool:
    """Validate credentials against the configured single admin user.

    This is intentionally a flat check rather than a database lookup -
    see docs/ARCHITECTURE.md for the migration path to a real user store
    (e.g. Postgres + a users table, or delegation to an external IdP)
    if this system grows beyond a single-operator internal tool.
    """
    if username != settings.ADMIN_USERNAME:
        return False
    return verify_password(password, settings.ADMIN_PASSWORD_HASH)
