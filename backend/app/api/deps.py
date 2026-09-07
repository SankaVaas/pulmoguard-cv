"""
Shared FastAPI dependencies, primarily authentication.

`get_current_user` is the single dependency every protected route uses.
It accepts credentials from EITHER source, in this order:
  1. The httpOnly session cookie (`AUTH_COOKIE_NAME`) - how the browser
     frontend authenticates, since its JS never has direct access to the token.
  2. A standard `Authorization: Bearer <token>` header - how API/CLI
     clients (curl, scripts, Swagger UI's "Authorize" button) authenticate.

Centralizing both paths here means the token-validation logic (and any
future changes, e.g. adding scope checks) lives in exactly one place,
regardless of which transport a given caller used.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import get_settings
from app.core.security import decode_access_token

settings = get_settings()

# auto_error=False: don't 401 just because the header is missing - the
# cookie might still be present. We do our own combined check below.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


def _extract_token(request: Request, header_token: str | None) -> str | None:
    cookie_token = request.cookies.get(settings.AUTH_COOKIE_NAME)
    return cookie_token or header_token


def get_current_user(request: Request, header_token: str | None = Depends(oauth2_scheme)) -> str:
    """Validate the token (cookie or bearer header) and return the
    authenticated username.

    Raises 401 Unauthorized (with a WWW-Authenticate header, per spec) if
    no token is present, or the token is expired/invalid.
    """
    token = _extract_token(request, header_token)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = decode_access_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username

