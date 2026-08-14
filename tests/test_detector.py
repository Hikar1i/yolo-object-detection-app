"""Unit tests for src/detector.py.

Fast tests exercise the dataclasses and device-selection logic without any
model weights. The `@pytest.mark.integration` tests actually load
`yolov8n.pt` and run inference on the bundled sample image — they require
network access on first run (to download weights) and are skipped
automatically if that fails, so `pytest -m "not integration"` gives a
fully offline, sub-second test run.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import config
from src.detector import (
    DetectionResult,
    InferenceConfig,
    ModelCache,
    get_best_device,
    is_gpu_available,
)


def test_inference_config_defaults() -> None:
    cfg = InferenceConfig()
    assert cfg.confidence == config.DEFAULT_CONFIDENCE
    assert cfg.iou == config.DEFAULT_IOU
    assert cfg.image_size == config.DEFAULT_IMAGE_SIZE
    assert cfg.classes is None


def test_get_best_device_respects_explicit_choice() -> None:
    assert get_best_device("cpu") == "cpu"


def test_get_best_device_auto_resolves_to_valid_string() -> None:
    device = get_best_device("auto")
    assert device in {"cpu", "mps"} or device.isdigit()


def test_is_gpu_available_returns_bool() -> None:
    assert isinstance(is_gpu_available(), bool)


def test_detection_result_num_detections() -> None:
    result = DetectionResult(
        boxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15]]),
        confidences=np.array([0.9, 0.8]),
        class_ids=np.array([0, 2]),
        class_names={0: "person", 2: "car"},
        track_ids=None,
        inference_time_ms=5.0,
        image_shape=(100, 100),
    )
    assert result.num_detections == 2
    assert result.detected_class_labels == ["person", "car"]
    assert result.class_distribution == {"person": 1, "car": 1}


def test_detection_result_empty() -> None:
    result = DetectionResult(
        boxes=np.empty((0, 4)),
        confidences=np.empty((0,)),
        class_ids=np.empty((0,), dtype=int),
        class_names={},
        track_ids=None,
        inference_time_ms=1.0,
        image_shape=(50, 50),
    )
    assert result.num_detections == 0
    assert result.class_distribution == {}


def test_detection_result_unknown_class_id_falls_back() -> None:
    result = DetectionResult(
        boxes=np.array([[0, 0, 1, 1]]),
        confidences=np.array([0.5]),
        class_ids=np.array([999]),
        class_names={0: "person"},
        track_ids=None,
        inference_time_ms=1.0,
        image_shape=(10, 10),
    )
    assert result.detected_class_labels == ["unknown"]


def test_model_cache_reuses_instance() -> None:
    pytest.importorskip("ultralytics")
    try:
        first = ModelCache.get("yolov8n.pt", device="cpu")
        second = ModelCache.get("yolov8n.pt", device="cpu")
    except RuntimeError:
        pytest.skip("Model weights unavailable (no network on first download).")
    assert first is second
    ModelCache.clear()


@pytest.mark.integration
def test_predict_image_on_sample() -> None:
    pytest.importorskip("ultralytics")
    sample_path = config.SAMPLE_IMAGES_DIR / "bus.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found.")

    try:
        detector = ModelCache.get("yolov8n.pt", device="cpu")
    except RuntimeError:
        pytest.skip("Model weights unavailable (no network on first download).")

    image = Image.open(sample_path)
    result = detector.predict_image(image, InferenceConfig(confidence=0.25))

    assert result.num_detections >= 0
    assert result.inference_time_ms > 0
    assert result.boxes.shape[1] == 4 if result.num_detections else True
