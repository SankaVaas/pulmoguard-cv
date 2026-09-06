"""
Shared FastAPI dependencies, primarily authentication.

`get_current_user` is the single dependency every protected route uses.
Centralizing it here means the token-validation logic (and any future
changes, e.g. adding scope checks) lives in exactly one place.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.security import decode_access_token

# tokenUrl is the path clients are told to POST credentials to when a 401
# is returned - purely documentational for the OpenAPI schema / Swagger UI.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Validate the bearer token and return the authenticated username.

    Raises 401 Unauthorized (with a WWW-Authenticate header, per spec) if
    the token is missing, expired, or invalid.
    """
    username = decode_access_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
