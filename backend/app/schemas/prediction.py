"""Pydantic schemas for the prediction endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(description="Model's predicted class label")
    confidence: float = Field(description="Max mean probability across MC-Dropout passes")
    normalized_entropy: float = Field(
        description="Predictive entropy normalized to [0, 1]; higher means less certain"
    )
    abstain: bool = Field(
        description="True if uncertainty exceeds the safe threshold; system recommends clinician review"
    )
    class_probabilities: dict[str, float] = Field(description="Full class probability distribution")
    mc_dropout_passes: int = Field(description="Number of stochastic forward passes used")
    filename: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    environment: str
    version: str
