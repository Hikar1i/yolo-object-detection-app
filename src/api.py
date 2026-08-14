"""Optional FastAPI REST API for headless YOLOv8 inference.

Run with:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Then POST an image to /predict for JSON detection results, or open
/docs for interactive Swagger documentation.
"""

from __future__ import annotations

import io
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from src import config
from src.detector import InferenceConfig, ModelCache
from src.utils import get_logger, validate_file_size

logger = get_logger(__name__)

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="REST API for running Ultralytics YOLOv8 object detection on uploaded images.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Detection(BaseModel):
    """A single detected object."""

    class_id: int
    class_name: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2]


class PredictionResponse(BaseModel):
    """Full response payload for a /predict call."""

    model: str
    image_width: int
    image_height: int
    inference_time_ms: float
    num_detections: int
    detections: list[Detection]


@app.get("/", tags=["Meta"])
def root() -> dict[str, str]:
    """Basic liveness/info endpoint."""
    return {
        "name": "YOLOv8 Object Detection API",
        "docs": "/docs",
        "predict_endpoint": "/predict",
        "health_endpoint": "/health",
    }


@app.get("/health", tags=["Meta"])
def health() -> dict[str, Any]:
    """Health check, including GPU availability."""
    from src.detector import get_device_info, is_gpu_available

    return {"status": "ok", "gpu_available": is_gpu_available(), "device": get_device_info()}


@app.get("/models", tags=["Meta"])
def list_models() -> dict[str, str]:
    """List available model identifiers keyed by their display label."""
    return config.AVAILABLE_MODELS


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Image file (jpg/png/bmp/webp)."),
    model: str = Query(default=config.DEFAULT_MODEL_NAME, description="Model weight identifier, e.g. yolov8n.pt"),
    confidence: float = Query(default=config.DEFAULT_CONFIDENCE, ge=0.0, le=1.0),
    iou: float = Query(default=config.DEFAULT_IOU, ge=0.0, le=1.0),
) -> PredictionResponse:
    """Run object detection on an uploaded image and return structured JSON.

    Raises:
        HTTPException(400): Invalid file type, oversized upload, or a
            corrupt/unreadable image.
        HTTPException(500): Model failed to load or run inference.
    """
    if model not in config.AVAILABLE_MODELS.values():
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. See GET /models.")

    contents = await file.read()
    valid, msg = validate_file_size(len(contents))
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    try:
        detector = ModelCache.get(model)
        cfg = InferenceConfig(confidence=confidence, iou=iou)
        result = detector.predict_image(image, cfg)
    except RuntimeError as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    detections = [
        Detection(
            class_id=int(cid),
            class_name=result.class_names.get(int(cid), "unknown"),
            confidence=float(conf),
            box=[round(float(v), 2) for v in box],
        )
        for box, cid, conf in zip(result.boxes, result.class_ids, result.confidences)
    ]

    return PredictionResponse(
        model=model,
        image_width=image.width,
        image_height=image.height,
        inference_time_ms=round(result.inference_time_ms, 2),
        num_detections=result.num_detections,
        detections=detections,
    )
