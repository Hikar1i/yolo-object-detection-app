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
    get_available_models,
    load_history,
    pil_to_bgr,
    save_history_record,
    save_uploaded_model,
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
    assert "文件过大" in msg


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


def test_upload_model_preserves_existing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    path = save_uploaded_model("自定义.pt", b"original weights")
    assert path.read_bytes() == b"original weights"
    with pytest.raises(FileExistsError):
        save_uploaded_model("自定义.pt", b"replacement weights")
    assert path.read_bytes() == b"original weights"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("filename,contents", [
    ("../outside.pt", b"weights"), ("dir/model.pt", b"weights"),
    ("dir\\model.pt", b"weights"), ("bad\x00.pt", b"weights"),
    ("model.onnx", b"weights"), ("model.pt", b""), ("", b"weights"),
])
def test_upload_model_rejects_invalid_input(tmp_path, monkeypatch, filename, contents) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    with pytest.raises(ValueError):
        save_uploaded_model(filename, contents)
    assert not list(tmp_path.iterdir())


def test_upload_model_checks_size_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "MAX_UPLOAD_SIZE_MB", 1)
    with pytest.raises(ValueError, match="文件过大"):
        save_uploaded_model("model.pt", b"a" * (1024 * 1024 + 1))
    assert not list(tmp_path.iterdir())


def test_available_models_include_local_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    local = save_uploaded_model("custom.PT", b"weights")
    (tmp_path / "incomplete.upload").write_bytes(b"partial")
    (tmp_path / "directory.pt").mkdir()
    (tmp_path / "link.pt").symlink_to(local)
    models = get_available_models()
    assert models == {**config.AVAILABLE_MODELS, "本地模型：custom.PT": str(local.resolve())}


def test_upload_model_cleans_up_after_publish_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    def denied(*args):
        raise PermissionError("permission denied")

    monkeypatch.setattr("src.utils.os.link", denied)
    with pytest.raises(PermissionError):
        save_uploaded_model("model.pt", b"weights")
    assert not list(tmp_path.iterdir())
