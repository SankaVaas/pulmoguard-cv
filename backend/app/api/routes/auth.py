"""
Authentication routes.

Implements the OAuth2 "password" grant (username + password -> bearer
token) against the single configured admin credential. See
core/security.py::authenticate_admin for the credential check and
docs/ARCHITECTURE.md for the path to a real multi-user store if needed.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.config import get_settings
from app.core.security import authenticate_admin, create_access_token
from app.schemas.auth import TokenResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
settings = get_settings()


@router.post("/token", response_model=TokenResponse)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """Exchange username/password for a bearer token.

    Uses the standard OAuth2 password-grant form encoding (application/x-www-form-urlencoded
    with `username` and `password` fields) so it works out of the box with
    Swagger UI's "Authorize" button and standard OAuth2 client libraries.
    """
    if not authenticate_admin(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=form_data.username)
    return TokenResponse(
        access_token=access_token,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )
