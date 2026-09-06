"""
Prediction route: the core clinical-triage endpoint.

Protected by JWT bearer auth (get_current_user). Accepts a single image
upload, runs MC-Dropout inference via the model service, and returns the
prediction along with the abstention decision and a human-readable message -
mirroring the standalone FastAPI app design from the ml/ layer, but now
behind authentication and with production logging/error handling.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_current_user
from app.schemas.prediction import PredictionResponse
from app.services.model_service import get_predictor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["prediction"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB - generous for a chest X-ray JPEG/PNG


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
) -> PredictionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        predictor = get_predictor()
        result = predictor.predict(image_bytes)
    except Exception as exc:
        logger.exception(f"Inference failed for user={current_user}, file={file.filename}")
        raise HTTPException(status_code=500, detail="Inference failed. See server logs.") from exc

    response = result.to_dict()
    response["filename"] = file.filename or "unknown"
    response["message"] = (
        "Model uncertainty exceeds the safe threshold for this case. "
        "Recommend referral to a radiologist rather than relying on this prediction."
        if response["abstain"]
        else "Prediction confidence within accepted operating range."
    )

    logger.info(
        f"Prediction served | user={current_user} | file={file.filename} | "
        f"predicted_class={response['predicted_class']} | abstain={response['abstain']}"
    )
    return PredictionResponse(**response)
