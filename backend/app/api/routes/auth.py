"""
Authentication routes.

Implements the OAuth2 "password" grant (username + password -> token)
against the single configured admin credential, plus session endpoints
(`/me`, `/logout`) that work off the httpOnly cookie set at login. See
core/security.py::authenticate_admin for the credential check and
docs/ARCHITECTURE.md §5.3 for the cookie-vs-bearer rationale, and §5.1 for
the path to a real multi-user store if needed.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.security import authenticate_admin, create_access_token
from app.schemas.auth import TokenResponse, UserResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
settings = get_settings()


def _set_auth_cookie(response: Response, token: str) -> None:
    """Attach the session cookie to a response.

    - httponly=True: never readable by page JavaScript (mitigates XSS token theft).
    - samesite="strict": cookie is withheld on genuinely cross-site requests;
      "localhost:5173" <-> "localhost:8000" (or two subdomains behind the
      reverse proxy in prod) are same-site by browser definition, so normal
      frontend<->backend calls are unaffected. See docs/ARCHITECTURE.md §5.3.
    - secure: required for a cookie to be sent over anything but localhost
      HTTP; forced on outside local development.
    """
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=token,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="strict",
        secure=settings.ENVIRONMENT != "development",
        path="/",
    )


@router.post("/token", response_model=TokenResponse)
@limiter.limit(f"{settings.AUTH_RATE_LIMIT_PER_MINUTE}/minute")
def login_for_access_token(
    request: Request, response: Response, form_data: OAuth2PasswordRequestForm = Depends()
) -> TokenResponse:
    """Exchange username/password for a token.

    Sets an httpOnly session cookie on the response (how the browser
    frontend authenticates from then on) AND returns the raw token in the
    JSON body (how API/CLI clients and Swagger UI's "Authorize" button
    authenticate - they never see the cookie). Both are valid; either is
    sufficient to call protected routes (see api/deps.py::get_current_user).

    Rate-limited more strictly than the API default (see
    AUTH_RATE_LIMIT_PER_MINUTE) since this is the endpoint a credential
    brute-force attempt would target. Both `request: Request` and
    `response: Response` are required by slowapi's decorator: `request` to
    check/track the limit, and `response` so it has somewhere to write the
    X-RateLimit-* headers onto, since this route returns a Pydantic model
    rather than a raw Response object.

    Uses the standard OAuth2 password-grant form encoding
    (application/x-www-form-urlencoded with `username` and `password`
    fields) so it works out of the box with Swagger UI and standard OAuth2
    client libraries.
    """
    if not authenticate_admin(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(subject=form_data.username)
    _set_auth_cookie(response, access_token)

    return TokenResponse(
        access_token=access_token,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: str = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated username.

    Used by the frontend on page load to check whether an existing session
    cookie is still valid, without needing to store or read the token
    itself in JavaScript.
    """
    return UserResponse(username=current_user)


@router.post("/logout")
def logout(response: Response) -> dict:
    """Clear the session cookie. Stateless JWTs can't be server-side
    revoked without a blocklist (out of scope here - see
    docs/ARCHITECTURE.md); this ends the *browser session*, which is the
    logout users actually expect from clicking "Sign out"."""
    response.delete_cookie(key=settings.AUTH_COOKIE_NAME, path="/")
    return {"detail": "Logged out"}
