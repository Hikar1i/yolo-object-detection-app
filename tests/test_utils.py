"""Unit tests for src/utils.py — pure functions, no model loading required."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src import config
from src.utils import (
    DetectionRecord,
    bgr_to_pil,
    draw_detections,
    format_bytes,
    get_box_color,
    load_history,
    pil_to_bgr,
    save_history_record,
    validate_file_size,
)


def test_pil_bgr_roundtrip_preserves_shape() -> None:
    img = Image.new("RGB", (64, 48), color=(10, 20, 30))
    bgr = pil_to_bgr(img)
    assert bgr.shape == (48, 64, 3)
    back = bgr_to_pil(bgr)
    assert back.size == (64, 48)


def test_pil_to_bgr_channel_order() -> None:
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))  # pure red
    bgr = pil_to_bgr(img)
    # Red in RGB -> (B=0, G=0, R=255) in BGR
    assert bgr[0, 0].tolist() == [0, 0, 255]


def test_get_box_color_is_deterministic_and_in_palette() -> None:
    color_a = get_box_color(3)
    color_b = get_box_color(3)
    assert color_a == color_b
    assert color_a in config.BOX_COLOR_PALETTE


def test_get_box_color_wraps_around_palette() -> None:
    n = len(config.BOX_COLOR_PALETTE)
    assert get_box_color(0) == get_box_color(n)


def test_draw_detections_returns_new_array_same_shape() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 50]])
    class_ids = np.array([0])
    confidences = np.array([0.9])
    class_names = {0: "person"}

    annotated = draw_detections(image, boxes, class_ids, confidences, class_names)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)  # something was drawn
    assert np.array_equal(image, np.zeros((100, 100, 3), dtype=np.uint8))  # original untouched


def test_draw_detections_handles_empty_boxes() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    annotated = draw_detections(image, np.empty((0, 4)), np.empty((0,)), np.empty((0,)), {})
    assert np.array_equal(annotated, image)


@pytest.mark.parametrize(
    "num_bytes,expected_unit",
    [(500, "B"), (2048, "KB"), (5 * 1024 * 1024, "MB"), (3 * 1024**3, "GB")],
)
def test_format_bytes_picks_correct_unit(num_bytes: int, expected_unit: str) -> None:
    assert format_bytes(num_bytes).endswith(expected_unit)


def test_validate_file_size_accepts_small_file() -> None:
    valid, msg = validate_file_size(1024)
    assert valid is True
    assert msg == ""


def test_validate_file_size_rejects_oversized_file() -> None:
    too_big = (config.MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024
    valid, msg = validate_file_size(too_big)
    assert valid is False
    assert "too large" in msg.lower()


def test_save_and_load_history_roundtrip(tmp_path, monkeypatch) -> None:
    history_file = tmp_path / "history.csv"
    monkeypatch.setattr(config, "HISTORY_FILE", history_file)

    record = DetectionRecord(
        timestamp=1700000000.0,
        source_type="image",
        model_name="yolov8n.pt",
        num_detections=3,
        inference_time_ms=12.5,
        fps=80.0,
        classes_detected=["person", "car", "person"],
    )
    save_history_record(record)
    rows = load_history()

    assert len(rows) == 1
    assert rows[0]["source_type"] == "image"
    assert rows[0]["num_detections"] == "3"
    assert rows[0]["classes_detected"] == "person|car|person"
