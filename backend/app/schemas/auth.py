"""Pydantic schemas for authentication endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int = Field(description="Token lifetime in minutes from issuance")


class TokenPayload(BaseModel):
    sub: str


class UserResponse(BaseModel):
    username: str
