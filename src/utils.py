"""Shared helper functions: logging, drawing, I/O, and analytics persistence."""

from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src import config

_LOGGER_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a configured module-level logger.

    Sets up a rotating console + file handler exactly once per process,
    then returns namespaced child loggers on every subsequent call.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    global _LOGGER_CONFIGURED
    root = logging.getLogger("yolov8_app")

    if not _LOGGER_CONFIGURED:
        root.setLevel(config.LOG_LEVEL)
        formatter = logging.Formatter(config.LOG_FORMAT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        try:
            file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            root.warning("Could not create log file at %s; logging to console only.", config.LOG_FILE)

        _LOGGER_CONFIGURED = True

    return root.getChild(name)


logger = get_logger(__name__)


@dataclass
class DetectionRecord:
    """A single detection event, ready to be persisted to the history log."""

    timestamp: float
    source_type: str  # "image" | "video" | "webcam"
    model_name: str
    num_detections: int
    inference_time_ms: float
    fps: float
    classes_detected: list[str] = field(default_factory=list)


def timer_ms() -> float:
    """Return a high-resolution monotonic timestamp in milliseconds."""
    return time.perf_counter() * 1000.0


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV-style BGR ``np.ndarray``."""
    rgb_array = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV-style BGR ``np.ndarray`` to a PIL RGB image."""
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL image to bytes for a Streamlit download button."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def get_box_color(class_id: int) -> tuple[int, int, int]:
    """Deterministically map a class id to an RGB color from the palette."""
    return config.BOX_COLOR_PALETTE[class_id % len(config.BOX_COLOR_PALETTE)]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    class_names: dict[int, str],
    track_ids: np.ndarray | None = None,
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes, labels, and confidence scores onto a BGR frame.

    Args:
        image: BGR frame to annotate (modified copy is returned, original
            is left untouched).
        boxes: ``(N, 4)`` array of ``[x1, y1, x2, y2]`` pixel coordinates.
        class_ids: ``(N,)`` array of integer class ids.
        confidences: ``(N,)`` array of detection confidence scores.
        class_names: Mapping of class id -> human-readable class name.
        track_ids: Optional ``(N,)`` array of persistent tracker ids.
        line_thickness: Bounding-box line thickness in pixels.

    Returns:
        A new annotated BGR frame.
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(annotated.shape[:2]) / 1000)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        class_id = int(class_ids[i])
        confidence = float(confidences[i])
        color = get_box_color(class_id)

        label = f"{class_names.get(class_id, 'unknown')} {confidence:.2f}"
        if track_ids is not None:
            label = f"#{int(track_ids[i])} {label}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        label_y1 = max(y1 - text_h - baseline, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, y1 - baseline),
            font, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def format_bytes(num_bytes: float) -> str:
    """Human-readable byte size, e.g. ``1536`` -> ``"1.50 KB"``."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def save_history_record(record: DetectionRecord) -> None:
    """Append a detection event to the CSV analytics log.

    Fails silently (with a logged warning) rather than crashing the UI —
    analytics history is a nice-to-have, not a critical path.
    """
    file_exists = config.HISTORY_FILE.exists()
    try:
        with open(config.HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["timestamp", "source_type", "model_name", "num_detections",
                     "inference_time_ms", "fps", "classes_detected"]
                )
            writer.writerow(
                [
                    record.timestamp,
                    record.source_type,
                    record.model_name,
                    record.num_detections,
                    round(record.inference_time_ms, 2),
                    round(record.fps, 2),
                    "|".join(record.classes_detected),
                ]
            )
    except OSError as exc:
        logger.warning("Failed to write detection history: %s", exc)


def load_history() -> list[dict[str, Any]]:
    """Load the detection history CSV as a list of row dicts, newest last."""
    if not config.HISTORY_FILE.exists():
        return []
    try:
        with open(config.HISTORY_FILE, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))[-config.MAX_HISTORY_ROWS :]
    except OSError as exc:
        logger.warning("Failed to read detection history: %s", exc)
        return []


def clear_history() -> None:
    """Delete the detection history file, if present."""
    try:
        Path(config.HISTORY_FILE).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to clear detection history: %s", exc)


def validate_file_size(size_bytes: int) -> tuple[bool, str]:
    """Check an uploaded file against the configured max size.

    Returns:
        ``(is_valid, message)`` — message is empty when valid.
    """
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"File too large ({format_bytes(size_bytes)}). Max allowed: {config.MAX_UPLOAD_SIZE_MB} MB."
    return True, ""
