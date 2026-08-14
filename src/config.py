"""Centralized configuration for the YOLOv8 Object Detection application.

All tunable constants, paths, and default settings live here so the rest of
the codebase never hardcodes "magic" values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
ASSETS_DIR: Final[Path] = BASE_DIR / "assets"
SAMPLE_IMAGES_DIR: Final[Path] = ASSETS_DIR / "sample_images"
SAMPLE_VIDEO_DIR: Final[Path] = ASSETS_DIR / "sample_video"
MODELS_DIR: Final[Path] = BASE_DIR / "models"
OUTPUTS_DIR: Final[Path] = BASE_DIR / "outputs"
LOGS_DIR: Final[Path] = BASE_DIR / "logs"

for _dir in (MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
# Ultralytics auto-downloads weights on first use, so only the identifier is
# required. Keys are the human-friendly labels shown in the UI.
AVAILABLE_MODELS: Final[dict[str, str]] = {
    "YOLOv8 Nano (fastest, least accurate)": "yolov8n.pt",
    "YOLOv8 Small": "yolov8s.pt",
    "YOLOv8 Medium": "yolov8m.pt",
    "YOLOv8 Large": "yolov8l.pt",
    "YOLOv8 X-Large (slowest, most accurate)": "yolov8x.pt",
}

DEFAULT_MODEL_KEY: Final[str] = "YOLOv8 Nano (fastest, least accurate)"
DEFAULT_MODEL_NAME: Final[str] = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]

# Approximate published benchmarks (COCO val, mAP50-95, single V100 GPU).
# Used only for the "Model Comparison" / "Documentation" tables — not
# measured at runtime.
MODEL_BENCHMARKS: Final[dict[str, dict[str, float | int]]] = {
    "yolov8n.pt": {"params_m": 3.2, "map50_95": 37.3, "speed_ms_cpu": 80.4, "speed_ms_gpu": 1.47},
    "yolov8s.pt": {"params_m": 11.2, "map50_95": 44.9, "speed_ms_cpu": 128.4, "speed_ms_gpu": 2.66},
    "yolov8m.pt": {"params_m": 25.9, "map50_95": 50.2, "speed_ms_cpu": 234.7, "speed_ms_gpu": 5.86},
    "yolov8l.pt": {"params_m": 43.7, "map50_95": 52.9, "speed_ms_cpu": 375.2, "speed_ms_gpu": 9.06},
    "yolov8x.pt": {"params_m": 68.2, "map50_95": 53.9, "speed_ms_cpu": 479.1, "speed_ms_gpu": 14.37},
}

# --------------------------------------------------------------------------- #
# Inference defaults
# --------------------------------------------------------------------------- #
DEFAULT_CONFIDENCE: Final[float] = 0.25
MIN_CONFIDENCE: Final[float] = 0.0
MAX_CONFIDENCE: Final[float] = 1.0

DEFAULT_IOU: Final[float] = 0.45  # NMS IoU threshold
MIN_IOU: Final[float] = 0.0
MAX_IOU: Final[float] = 1.0

DEFAULT_IMAGE_SIZE: Final[int] = 640
SUPPORTED_IMAGE_SIZES: Final[list[int]] = [320, 480, 640, 960, 1280]

MAX_DETECTIONS: Final[int] = 300

# --------------------------------------------------------------------------- #
# Video / webcam
# --------------------------------------------------------------------------- #
DEFAULT_TRACKER: Final[str] = "bytetrack.yaml"
AVAILABLE_TRACKERS: Final[list[str]] = ["bytetrack.yaml", "botsort.yaml"]
VIDEO_FRAME_SKIP: Final[int] = 1  # process every Nth frame (1 = every frame)
WEBCAM_DEFAULT_INDEX: Final[int] = 0

# --------------------------------------------------------------------------- #
# File support
# --------------------------------------------------------------------------- #
SUPPORTED_IMAGE_TYPES: Final[list[str]] = ["jpg", "jpeg", "png", "bmp", "webp"]
SUPPORTED_VIDEO_TYPES: Final[list[str]] = ["mp4", "avi", "mov", "mkv"]
MAX_UPLOAD_SIZE_MB: Final[int] = 200

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
APP_TITLE: Final[str] = "YOLOv8 Object Detection"
APP_ICON: Final[str] = "🎯"
PAGE_LAYOUT: Final[str] = "wide"

BOX_COLOR_PALETTE: Final[list[tuple[int, int, int]]] = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE: Final[Path] = LOGS_DIR / "app.log"

# --------------------------------------------------------------------------- #
# Analytics / history
# --------------------------------------------------------------------------- #
HISTORY_FILE: Final[Path] = OUTPUTS_DIR / "detection_history.csv"
MAX_HISTORY_ROWS: Final[int] = 5000
