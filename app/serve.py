"""
FastAPI serving app for PulmoGuard.

Run locally after training (weights copied from Colab):
    uvicorn app.serve:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST -F "file=@sample_xray.jpeg" http://localhost:8000/predict

Endpoints:
    GET  /health   - liveness check
    POST /predict  - upload a chest X-ray image, receive triage decision

This app loads the model ONCE at startup (not per-request), which matters
for latency and for correctness of MC-Dropout batching semantics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running uvicorn from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from pulmoguard.infer import PulmoGuardPredictor  # noqa: E402
from pulmoguard.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

CHECKPOINT_PATH = os.environ.get(
    "PULMOGUARD_CHECKPOINT", "outputs/checkpoints/pulmoguard_best.pt"
)
CONFIG_PATH = os.environ.get("PULMOGUARD_CONFIG", "configs/config.yaml")

app = FastAPI(
    title="PulmoGuard API",
    description="Uncertainty-aware chest X-ray pneumonia triage with calibrated abstention.",
    version="1.0.0",
)

_predictor: PulmoGuardPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    """Load the model once when the server starts, failing fast and loudly
    if the checkpoint is missing rather than erroring on the first request."""
    global _predictor
    try:
        _predictor = PulmoGuardPredictor(
            checkpoint_path=CHECKPOINT_PATH, config_path=CONFIG_PATH
        )
        logger.info("Model loaded successfully. PulmoGuard API is ready.")
    except FileNotFoundError as exc:
        logger.error(
            f"Startup failed: {exc}\n"
            f"Train a model (notebooks/train_colab.ipynb), download the "
            f".pt checkpoint, and place it at '{CHECKPOINT_PATH}' "
            f"(or set PULMOGUARD_CHECKPOINT env var)."
        )
        raise


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server startup logs.")

    allowed_types = {"image/jpeg", "image/png", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {allowed_types}",
        )

    try:
        image_bytes = await file.read()
        result = _predictor.predict(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Prediction failed for {file.filename}: {exc}")
        raise HTTPException(status_code=500, detail="Inference failed. See server logs.") from exc

    response = result.to_dict()
    response["filename"] = file.filename

    if response["abstain"]:
        response["message"] = (
            "Model uncertainty exceeds the safe threshold for this case. "
            "Recommend referral to a radiologist rather than relying on this prediction."
        )
    else:
        response["message"] = "Prediction confidence within accepted operating range."

    return JSONResponse(content=response)
